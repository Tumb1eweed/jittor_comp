import os
from pathlib import Path

os.environ.setdefault("nvcc_path", "")
import numpy as np
import jittor as jt
from jittor import nn

from models.feature import FeatureExtraction
from models.utils import farthest_point_sampling_jt, knn_points


class PGDModel(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.feature_nets = FeatureExtraction(d_in=0, use_codebook=True)
        self.two_stage = bool(getattr(args, "pgd_two_stage", False) if args is not None else False)
        self.use_separate_stage2 = bool(
            getattr(args, "pgd_use_separate_stage2", False) if args is not None else False
        )
        if self.use_separate_stage2:
            self.feature_nets_stage2 = FeatureExtraction(d_in=0, use_codebook=True)
        self.second_stage_scale = float(getattr(args, "pgd_second_stage_scale", 1.0) if args is not None else 1.0)
        self.second_stage_tangent_only = bool(
            getattr(args, "pgd_second_stage_tangent_only", False) if args is not None else False
        )
        self.second_stage_tangent_scale = float(
            getattr(args, "pgd_second_stage_tangent_scale", 1.0) if args is not None else 1.0
        )
        self.second_stage_normal_scale = float(
            getattr(args, "pgd_second_stage_normal_scale", 0.15) if args is not None else 0.15
        )
        self.second_stage_surface_k = int(
            getattr(args, "pgd_second_stage_surface_k", 16) if args is not None else 16
        )
        self.use_stage2_dual_gate = bool(
            getattr(args, "pgd_use_stage2_dual_gate", False) if args is not None else False
        )
        self.stage2_dual_gate_scale = float(
            getattr(args, "pgd_stage2_dual_gate_scale", 0.90) if args is not None else 0.90
        )
        self.use_refine_gate = bool(getattr(args, "pgd_use_refine_gate", False) if args is not None else False)
        self.refine_gate_scale = float(getattr(args, "pgd_refine_gate_scale", 0.25) if args is not None else 0.25)
        self.use_noise_conditioning = bool(
            getattr(args, "pgd_use_noise_conditioning", False) if args is not None else False
        )
        self.noise_condition_scale = float(
            getattr(args, "pgd_noise_condition_scale", 0.50) if args is not None else 0.50
        )
        self.noise_condition_min = float(
            getattr(args, "pgd_noise_condition_min", 0.005) if args is not None else 0.005
        )
        self.noise_condition_max = float(
            getattr(args, "pgd_noise_condition_max", 0.020) if args is not None else 0.020
        )
        self.detach_second_stage_backbone = bool(
            getattr(args, "pgd_train_detach_second_stage_backbone", False) if args is not None else False
        )
        self.detach_second_stage_features = bool(
            getattr(args, "pgd_train_detach_second_stage_features", False) if args is not None else False
        )
        self.use_surface_flow = bool(
            getattr(args, "pgd_use_surface_flow", False) if args is not None else False
        )
        self.use_surface_head = bool(
            getattr(args, "pgd_use_surface_head", False) if args is not None else False
        )
        self.use_surface_vector_head = bool(
            getattr(args, "pgd_use_surface_vector_head", False) if args is not None else False
        )
        self.surface_head_max_distance = float(
            getattr(args, "pgd_surface_head_max_distance", 0.02) if args is not None else 0.02
        )
        self.surface_vector_max_distance = float(
            getattr(args, "pgd_surface_vector_max_distance", 0.02) if args is not None else 0.02
        )
        self.surface_vector_unit_slope = bool(
            getattr(args, "pgd_surface_vector_unit_slope", False) if args is not None else False
        )
        self.surface_flow_log_scale_min = float(
            getattr(args, "pgd_surface_flow_log_scale_min", -2.0) if args is not None else -2.0
        )
        self.surface_flow_log_scale_max = float(
            getattr(args, "pgd_surface_flow_log_scale_max", 0.4) if args is not None else 0.4
        )
        if self.use_refine_gate:
            self.refine_gate_fc1 = nn.Linear(6, 32)
            self.refine_gate_fc2 = nn.Linear(32, 1)
            self.refine_gate_fc2.weight.assign(jt.zeros_like(self.refine_gate_fc2.weight))
            if self.refine_gate_fc2.bias is not None:
                self.refine_gate_fc2.bias.assign(jt.zeros_like(self.refine_gate_fc2.bias))
        if self.use_stage2_dual_gate:
            # All inputs are rotation-invariant scalars.  Zero-initialising the
            # final layer makes both multipliers exactly one, so enabling this
            # module reproduces the submitted checkpoint before fine-tuning.
            self.stage2_dual_gate_fc1 = nn.Linear(6, 32)
            self.stage2_dual_gate_fc2 = nn.Linear(32, 2)
            self.stage2_dual_gate_fc2.weight.assign(jt.zeros_like(self.stage2_dual_gate_fc2.weight))
            if self.stage2_dual_gate_fc2.bias is not None:
                self.stage2_dual_gate_fc2.bias.assign(jt.zeros_like(self.stage2_dual_gate_fc2.bias))
        if self.use_noise_conditioning:
            hidden_dim = int(
                getattr(args, "pgd_noise_condition_hidden_dim", 16) if args is not None else 16
            )
            self.noise_condition_fc1 = nn.Linear(2, hidden_dim)
            self.noise_condition_fc2 = nn.Linear(hidden_dim, 2)
            # Enabling the conditioner on an old checkpoint is initially an
            # exact identity.  Output 0 controls stage 1 and output 1 stage 2.
            self.noise_condition_fc2.weight.assign(
                jt.zeros_like(self.noise_condition_fc2.weight)
            )
            if self.noise_condition_fc2.bias is not None:
                self.noise_condition_fc2.bias.assign(
                    jt.zeros_like(self.noise_condition_fc2.bias)
                )
        if self.use_surface_flow:
            # StraightPCF predicts a patch-level remaining-distance factor in
            # addition to its velocity field.  The final layer is exactly zero
            # at construction, hence exp(log_scale)=1 and an old PGD
            # checkpoint retains bitwise-equivalent displacement before
            # fine-tuning.
            pooled_dim = 2 * int(self.feature_nets.feature_dim)
            hidden_dim = int(
                getattr(args, "pgd_surface_flow_hidden_dim", 32) if args is not None else 32
            )
            self.surface_flow_distance_fc1 = nn.Linear(pooled_dim, hidden_dim)
            self.surface_flow_distance_fc2 = nn.Linear(hidden_dim, 1)
            self.surface_flow_distance_fc2.weight.assign(
                jt.zeros_like(self.surface_flow_distance_fc2.weight)
            )
            if self.surface_flow_distance_fc2.bias is not None:
                self.surface_flow_distance_fc2.bias.assign(
                    jt.zeros_like(self.surface_flow_distance_fc2.bias)
                )
        if self.use_surface_head:
            # Predict an explicit local surface frame and a signed point-to-
            # surface correction.  Only the distance layer is zero initialised:
            # an existing checkpoint therefore starts with exactly the same
            # output, while the normal branch can receive direct supervision.
            hidden_dim = int(
                getattr(args, "pgd_surface_head_hidden_dim", 64) if args is not None else 64
            )
            feature_dim = int(self.feature_nets.feature_dim)
            self.surface_head_fc1 = nn.Linear(feature_dim, hidden_dim)
            self.surface_head_normal_fc2 = nn.Linear(hidden_dim, 3)
            self.surface_head_distance_fc2 = nn.Linear(hidden_dim, 1)
            self.surface_head_distance_fc2.weight.assign(
                jt.zeros_like(self.surface_head_distance_fc2.weight)
            )
            if self.surface_head_distance_fc2.bias is not None:
                self.surface_head_distance_fc2.bias.assign(
                    jt.zeros_like(self.surface_head_distance_fc2.bias)
                )
        if self.use_surface_vector_head:
            # Directly predicting the residual vector avoids the sign
            # ambiguity of normal * signed-distance.  Concatenating both
            # stage displacements exposes how much correction has already
            # happened.  A zero final layer preserves the old checkpoint
            # exactly before training.
            hidden_dim = int(
                getattr(args, "pgd_surface_vector_hidden_dim", 64)
                if args is not None else 64
            )
            feature_dim = int(self.feature_nets.feature_dim) + 6
            self.surface_vector_fc1 = nn.Linear(feature_dim, hidden_dim)
            self.surface_vector_fc2 = nn.Linear(hidden_dim, 3)
            self.surface_vector_fc2.weight.assign(
                jt.zeros_like(self.surface_vector_fc2.weight)
            )
            if self.surface_vector_fc2.bias is not None:
                self.surface_vector_fc2.bias.assign(
                    jt.zeros_like(self.surface_vector_fc2.bias)
                )

    @classmethod
    def load_from_npz(cls, path, args=None):
        model = cls(args=args)
        model.load_npz(path)
        model.eval()
        return model

    def load_npz(self, path):
        params = np.load(path)
        state = self.state_dict()
        loaded = 0
        missing = []
        for name, var in state.items():
            candidates = [name]
            if name.startswith("feature_nets."):
                candidates.append(name[len("feature_nets."):])
            elif name.startswith("feature_nets_stage2."):
                suffix = name[len("feature_nets_stage2."):]
                # Old checkpoints have one shared backbone.  Initialising the
                # new stage-2 copy from that backbone preserves the exact
                # two-stage function before specialization.
                candidates.extend(["feature_nets." + suffix, suffix])
            source_name = None
            for candidate in candidates:
                if candidate in params:
                    source_name = candidate
                    break
            if source_name is None:
                missing.append(name)
                continue
            arr = params[source_name]
            if tuple(arr.shape) != tuple(var.shape):
                raise ValueError(f"shape mismatch for {name}: npz {arr.shape}, model {tuple(var.shape)}")
            var.assign(jt.array(arr))
            loaded += 1
        self._load_report = {"path": str(path), "loaded": loaded, "missing": missing}
        return self._load_report

    def save_npz(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {k: v.numpy() for k, v in self.feature_nets.state_dict().items()}
        for name, var in self.state_dict().items():
            if name.startswith("feature_nets."):
                continue
            arrays[name] = var.numpy()
        np.savez_compressed(path, **arrays)

    def _refinement_gate(self, disp1, raw_disp2):
        gate = jt.ones((disp1.shape[0], disp1.shape[1], 1), dtype=jt.float32)
        if self.use_refine_gate:
            features = jt.concat([disp1, raw_disp2], dim=-1)
            raw_gate = self.refine_gate_fc2(nn.relu(self.refine_gate_fc1(features)))
            gate = gate * (1.0 + self.refine_gate_scale * jt.tanh(raw_gate))
        return gate

    def _noise_condition_gates(self, noise_std, batch_size):
        ones = jt.ones((batch_size, 1, 1), dtype=jt.float32)
        if not self.use_noise_conditioning:
            return ones, ones

        if noise_std is None:
            midpoint = 0.5 * (self.noise_condition_min + self.noise_condition_max)
            sigma = jt.ones((batch_size, 1), dtype=jt.float32) * midpoint
        elif isinstance(noise_std, jt.Var):
            sigma = noise_std.float32()
        else:
            sigma = jt.array(np.asarray(noise_std, dtype=np.float32))

        if len(sigma.shape) == 0:
            sigma = jt.ones((batch_size, 1), dtype=jt.float32) * sigma
        elif len(sigma.shape) == 1:
            sigma = sigma.unsqueeze(1)
        if sigma.shape[0] == 1 and batch_size > 1:
            sigma = jt.ones((batch_size, 1), dtype=jt.float32) * sigma[0:1, :]
        if sigma.shape[0] != batch_size:
            raise ValueError(
                "noise_std batch {} does not match point batch {}".format(
                    sigma.shape[0], batch_size
                )
            )

        midpoint = 0.5 * (self.noise_condition_min + self.noise_condition_max)
        half_range = max(0.5 * (self.noise_condition_max - self.noise_condition_min), 1e-8)
        z = (sigma[:, 0:1] - midpoint) / half_range
        z = jt.minimum(jt.maximum(z, -1.5), 1.5)
        features = jt.concat([z, z * z], dim=1)
        logits = self.noise_condition_fc2(
            nn.relu(self.noise_condition_fc1(features))
        )
        gates = 1.0 + self.noise_condition_scale * jt.tanh(logits)
        return gates[:, 0:1].unsqueeze(1), gates[:, 1:2].unsqueeze(1)

    def _surface_flow_distance(self, point_features):
        pooled = jt.concat(
            [point_features.mean(dim=1), point_features.max(dim=1)], dim=1
        )
        log_scale = self.surface_flow_distance_fc2(
            nn.relu(self.surface_flow_distance_fc1(pooled))
        )
        log_scale = jt.minimum(
            jt.maximum(log_scale, self.surface_flow_log_scale_min),
            self.surface_flow_log_scale_max,
        )
        return jt.exp(log_scale).unsqueeze(1), log_scale.unsqueeze(1)

    def _surface_head(self, point_features):
        hidden = nn.relu(self.surface_head_fc1(point_features))
        normal_raw = self.surface_head_normal_fc2(hidden)
        normal = normal_raw / (jt.norm(normal_raw, dim=-1, keepdims=True) + 1e-8)
        distance_raw = self.surface_head_distance_fc2(hidden)
        distance = self.surface_head_max_distance * jt.tanh(distance_raw)
        correction = distance * normal
        return normal, distance, correction

    def _surface_vector_head(self, point_features, disp1, disp2):
        features = jt.concat([point_features, disp1, disp2], dim=-1)
        hidden = nn.relu(self.surface_vector_fc1(features))
        raw = self.surface_vector_fc2(hidden)
        if self.surface_vector_unit_slope:
            # Keep the correction bounded while making the derivative at zero
            # equal to one.  max_distance*tanh(raw) has slope max_distance
            # (0.02 here), which made a zero-initialised residual head learn
            # roughly 50x too slowly.
            return self.surface_vector_max_distance * jt.tanh(
                raw / self.surface_vector_max_distance
            )
        return self.surface_vector_max_distance * jt.tanh(raw)

    def _local_surface_geometry(self, points):
        """Estimate PCA normals from the current point set only.

        This is used solely to decompose the second-stage displacement.  It
        does not access clean points or mesh normals and is therefore valid at
        test time.  The graph includes the point itself, which is discarded
        before the covariance fit.
        """
        k = max(2, int(self.second_stage_surface_k))
        distances, _, neighbours = knn_points(
            points, points, k=k + 1, return_nn=True
        )
        neighbours = neighbours[:, :, 1:, :]
        local_spacing = jt.sqrt(jt.maximum(distances[:, :, 1:].mean(dim=2, keepdims=True), 1e-12))
        # A covariance eigensolve is unnecessarily fragile in Jittor's CUDA
        # runtime (and expensive for every patch).  Two independent local
        # chords provide the same tangent-plane normal up to first order.  Use
        # the strongest of three chord pairs to avoid nearly collinear picks.
        base = points.unsqueeze(2)
        v0 = neighbours[:, :, 0:1, :] - base
        v1 = neighbours[:, :, 1:2, :] - base
        v2 = neighbours[:, :, 2:3, :] - base
        c01 = jt.cross(v0, v1, dim=-1)[:, :, 0, :]
        c12 = jt.cross(v1, v2, dim=-1)[:, :, 0, :]
        c20 = jt.cross(v2, v0, dim=-1)[:, :, 0, :]
        n01 = jt.norm(c01, dim=-1, keepdims=True)
        n12 = jt.norm(c12, dim=-1, keepdims=True)
        n20 = jt.norm(c20, dim=-1, keepdims=True)
        use12 = (n12 > n01).float32()
        best = c01 * (1.0 - use12) + c12 * use12
        best_norm = n01 * (1.0 - use12) + n12 * use12
        use20 = (n20 > best_norm).float32()
        best = best * (1.0 - use20) + c20 * use20
        normals = best / (jt.norm(best, dim=-1, keepdims=True) + 1e-8)
        return normals, local_spacing

    def _stage2_dual_gates(
        self, disp1, raw_disp2, normal_disp2, tangent_disp2, local_spacing
    ):
        disp1_mag = jt.norm(disp1, dim=-1, keepdims=True)
        raw2_mag = jt.norm(raw_disp2, dim=-1, keepdims=True)
        normal_mag = jt.norm(normal_disp2, dim=-1, keepdims=True)
        tangent_mag = jt.norm(tangent_disp2, dim=-1, keepdims=True)
        alignment = jt.abs(
            jt.sum(disp1 * raw_disp2, dim=-1, keepdims=True)
            / (disp1_mag * raw2_mag + 1e-8)
        )
        features = jt.concat(
            [
                disp1_mag,
                raw2_mag,
                normal_mag,
                tangent_mag,
                local_spacing,
                alignment,
            ],
            dim=-1,
        )
        logits = self.stage2_dual_gate_fc2(
            nn.relu(self.stage2_dual_gate_fc1(features))
        )
        scale = float(self.stage2_dual_gate_scale)
        gates = 1.0 + scale * jt.tanh(logits)
        return gates[:, :, 0:1], gates[:, :, 1:2]

    def execute(self, pcl_noisy, noise_std=None, category_id=None, return_dict=False):
        b, n, _ = pcl_noisy.shape
        offset = jt.array(np.array([(i + 1) * n for i in range(b)], dtype=np.int32))
        stage1_noise_gate, stage2_noise_gate = self._noise_condition_gates(noise_std, b)
        surface_features = None
        if (
            self.use_surface_flow
            or ((self.use_surface_head or self.use_surface_vector_head) and not self.two_stage)
        ):
            surface_features = self.feature_nets(
                pcl_noisy, None, offset, return_features=True
            )
            raw_disp = self.feature_nets.project_features(surface_features)
        else:
            raw_disp = self.feature_nets(pcl_noisy, None, offset)
        disp = raw_disp * stage1_noise_gate
        if self.two_stage:
            x1 = pcl_noisy + disp
            stage2_features = None
            stage2_net = (
                self.feature_nets_stage2
                if self.use_separate_stage2
                else self.feature_nets
            )
            if self.detach_second_stage_features:
                # Head-only stage-2 specialization needs gradients through the
                # copied output MLP, but not through the expensive point
                # encoder/decoder. Detaching here keeps that exact boundary.
                with jt.no_grad():
                    stage2_features = stage2_net(
                        x1.detach(), None, offset, return_features=True
                    )
                stage2_features = stage2_features.detach()
                raw_disp2 = stage2_net.project_features(stage2_features)
            elif self.detach_second_stage_backbone:
                with jt.no_grad():
                    if self.use_surface_head or self.use_surface_vector_head:
                        stage2_features = stage2_net(
                            x1.detach(), None, offset, return_features=True
                        )
                        raw_disp2 = stage2_net.project_features(stage2_features)
                    else:
                        raw_disp2 = stage2_net(x1.detach(), None, offset)
            else:
                if self.use_surface_head or self.use_surface_vector_head:
                    stage2_features = stage2_net(
                        x1, None, offset, return_features=True
                    )
                    raw_disp2 = stage2_net.project_features(stage2_features)
                else:
                    raw_disp2 = stage2_net(x1, None, offset)
            raw_disp2 = raw_disp2 * stage2_noise_gate
            refine_gate = self._refinement_gate(disp, raw_disp2)
            stage2_normal = None
            stage2_tangent = None
            normal_gate = None
            tangent_gate = None
            if self.second_stage_tangent_only or self.use_stage2_dual_gate:
                normals, local_spacing = self._local_surface_geometry(x1)
                normal_disp2 = jt.sum(raw_disp2 * normals, dim=-1, keepdims=True) * normals
                tangent_disp2 = raw_disp2 - normal_disp2
                if self.use_stage2_dual_gate:
                    normal_gate, tangent_gate = self._stage2_dual_gates(
                        disp, raw_disp2, normal_disp2, tangent_disp2, local_spacing
                    )
                    normal_disp2 = normal_disp2 * normal_gate
                    tangent_disp2 = tangent_disp2 * tangent_gate
                    raw_disp2 = normal_disp2 + tangent_disp2
                else:
                    # Compatibility path for the earlier inference-only hard
                    # decomposition experiment.
                    normal_disp2 = self.second_stage_normal_scale * normal_disp2
                    tangent_disp2 = self.second_stage_tangent_scale * tangent_disp2
                    raw_disp2 = tangent_disp2 + normal_disp2
                stage2_normal = normal_disp2 * refine_gate * self.second_stage_scale
                stage2_tangent = tangent_disp2 * refine_gate * self.second_stage_scale
            disp2 = raw_disp2 * refine_gate * self.second_stage_scale
            total_velocity = disp + disp2
            flow_distance = None
            flow_log_distance = None
            if self.use_surface_flow:
                flow_distance, flow_log_distance = self._surface_flow_distance(
                    surface_features
                )
                total_disp = total_velocity * flow_distance
            else:
                total_disp = total_velocity
            surface_base_disp = total_disp
            surface_head_normal = None
            surface_head_distance = None
            surface_head_correction = None
            if self.use_surface_head:
                surface_head_normal, surface_head_distance, surface_head_correction = (
                    self._surface_head(stage2_features)
                )
                total_disp = total_disp + surface_head_correction
            surface_vector_correction = None
            if self.use_surface_vector_head:
                surface_vector_correction = self._surface_vector_head(
                    stage2_features, disp, disp2
                )
                total_disp = total_disp + surface_vector_correction
            final = pcl_noisy + total_disp
            if return_dict:
                return {
                    "disp": total_disp,
                    "flow_velocity": total_velocity,
                    "flow_distance": flow_distance,
                    "flow_log_distance": flow_log_distance,
                    "surface_head_normal": surface_head_normal,
                    "surface_head_distance": surface_head_distance,
                    "surface_head_correction": surface_head_correction,
                    "surface_vector_correction": surface_vector_correction,
                    "surface_base_disp": surface_base_disp,
                    "raw_disp": raw_disp,
                    "raw_disp1": raw_disp,
                    "raw_disp2": raw_disp2,
                    "disp1": disp,
                    "disp2": disp2,
                    "disp2_raw_projected": raw_disp2,
                    "disp2_normal": stage2_normal,
                    "disp2_tangent": stage2_tangent,
                    "stage2_normal_gate": normal_gate,
                    "stage2_tangent_gate": tangent_gate,
                    "x1": x1,
                    "final": final,
                    "refine_gate": refine_gate,
                    "stage1_noise_gate": stage1_noise_gate,
                    "stage2_noise_gate": stage2_noise_gate,
                }
            return total_disp
        flow_distance = None
        flow_log_distance = None
        if self.use_surface_flow:
            flow_distance, flow_log_distance = self._surface_flow_distance(
                surface_features
            )
            velocity = disp
            disp = velocity * flow_distance
        surface_base_disp = disp
        surface_head_normal = None
        surface_head_distance = None
        surface_head_correction = None
        if self.use_surface_head:
            surface_head_normal, surface_head_distance, surface_head_correction = (
                self._surface_head(surface_features)
            )
            disp = disp + surface_head_correction
        surface_vector_correction = None
        if self.use_surface_vector_head:
            surface_vector_correction = self._surface_vector_head(
                surface_features, disp, jt.zeros_like(disp)
            )
            disp = disp + surface_vector_correction
        if return_dict:
            return {
                "disp": disp,
                "raw_disp": raw_disp,
                "flow_velocity": velocity if self.use_surface_flow else disp,
                "flow_distance": flow_distance,
                "flow_log_distance": flow_log_distance,
                "surface_head_normal": surface_head_normal,
                "surface_head_distance": surface_head_distance,
                "surface_head_correction": surface_head_correction,
                "surface_vector_correction": surface_vector_correction,
                "surface_base_disp": surface_base_disp,
                "stage1_noise_gate": stage1_noise_gate,
                "stage2_noise_gate": stage2_noise_gate,
            }
        return disp

    def denoise_langevin_dynamics(self, pcl_noisy, noise_std=None, category_id=None):
        pred_disp = self(pcl_noisy, noise_std=noise_std, category_id=category_id)
        return pcl_noisy + pred_disp

    def patch_based_denoise(
        self,
        pcl_noisy,
        patch_size=1000,
        seed_k=5,
        seed_k_alpha=10,
        patch_batch_size=None,
        fusion="select",
        noise_std=None,
        category_id=None,
        context_patch_size=None,
    ):
        pcl = pcl_noisy if isinstance(pcl_noisy, jt.Var) else jt.array(np.asarray(pcl_noisy, dtype=np.float32))
        assert len(pcl.shape) == 2 and pcl.shape[1] == 3
        n = pcl.shape[0]
        core_patch_size = int(patch_size)
        context_patch_size = (
            core_patch_size
            if context_patch_size is None
            else int(context_patch_size)
        )
        if context_patch_size < core_patch_size:
            raise ValueError("context_patch_size must be >= patch_size")
        if context_patch_size > n:
            raise ValueError("context_patch_size must be <= point count")
        num_patches = int(seed_k * n / patch_size)
        seed = farthest_point_sampling_jt(pcl.unsqueeze(0), num_patches)[0]
        dists, idx, patches = knn_points(
            seed.unsqueeze(0),
            pcl.unsqueeze(0),
            k=context_patch_size,
            return_nn=True,
        )
        # kNN output is distance-sorted. The nearest core keeps the original
        # fusion contract; outer points supply context only and are discarded
        # after the network forward.
        patch_dists = dists[0, :, :core_patch_size]
        point_idxs = idx[0, :, :core_patch_size].int64()
        patches_centered = patches[0] - seed[:, None, :]
        denom = jt.maximum(patch_dists[:, -1:], jt.array(1e-12, dtype=jt.float32))
        patch_dists = patch_dists / denom
        all_dists = jt.full((num_patches, n), 1e10, dtype=jt.float32).scatter(1, point_idxs, patch_dists)
        best_patch = jt.argmax(-all_dists, dim=0)[0].int64()

        patches_denoised = []
        i = 0
        patch_step = int(n / (seed_k_alpha * patch_size))
        if patch_batch_size is not None:
            if context_patch_size > core_patch_size:
                patch_step = max(1, int(patch_batch_size))
            else:
                patch_step = max(patch_step, int(patch_batch_size))
        if patch_step <= 0:
            raise ValueError("Seed_k_alpha needs to be decreased to increase patch_step")
        if fusion not in ("select", "weighted"):
            raise ValueError("unsupported patch fusion: {}".format(fusion))
        while i < num_patches:
            curr = patches_centered[i:i + patch_step]
            den = self.denoise_langevin_dynamics(curr, noise_std=noise_std, category_id=category_id)
            patches_denoised.append(den[:, :core_patch_size, :])
            i += patch_step
        patches_denoised = jt.concat(patches_denoised, dim=0) + seed[:, None, :]
        if fusion == "weighted":
            # Every output point remains associated with exactly one input
            # point.  Unlike concatenating overlap patches, this cannot
            # change cardinality or create duplicate points at patch seams.
            # A smooth centre-biased window suppresses unreliable patch-edge
            # predictions and replaces select-fusion's hard Voronoi boundary.
            window = jt.maximum(1.0 - patch_dists, 0.0) ** 2 + 1e-4
            weighted = patches_denoised * window.unsqueeze(-1)
            numerator = weighted.reindex_reduce(
                "add", [n, 3], ["@e0(i0,i1)", "i2"], extras=[point_idxs]
            )
            denominator = window.reindex_reduce(
                "add", [n], ["@e0(i0,i1)"], extras=[point_idxs]
            )
            return (numerator / jt.maximum(denominator.unsqueeze(-1), 1e-12)).float32()
        local_ids = jt.arange(core_patch_size).reshape(1, core_patch_size).broadcast((num_patches, core_patch_size)).int64()
        local_for_point = jt.zeros((num_patches, n), dtype=jt.int64).scatter(1, point_idxs, local_ids)
        point_ids = jt.arange(n).int64()
        selected_local = local_for_point.reshape(-1)[best_patch * n + point_ids]
        out = patches_denoised.reshape(num_patches * core_patch_size, 3)[
            best_patch * core_patch_size + selected_local, :
        ]
        return out.float32()
