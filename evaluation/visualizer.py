"""Multi-camera telemetry rollout visualizer (GIF and 6-keyframe strip) for AlloyFlow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from envs.base import CAMERA_NAMES, TASK_SPECS, norm_gripper_to_raw
from envs.sim_env import SimEnv


@dataclass
class RolloutStepRecord:
    """Per-step snapshot captured during a closed-loop evaluation episode."""

    step_idx: int
    max_steps: int
    task_id: int
    episode_label: str
    domain: str
    cam_images: dict[str, np.ndarray]
    keypoints: dict[str, np.ndarray]
    camera_weights: np.ndarray
    pred_traj_2d: dict[str, tuple[np.ndarray, np.ndarray]]
    gt_traj_2d: dict[str, tuple[np.ndarray, np.ndarray]] | None
    src_marker_2d: dict[str, tuple[np.ndarray, np.ndarray]]
    tgt_marker_2d: dict[str, tuple[np.ndarray, np.ndarray]]
    proprio: np.ndarray
    gt_proprio: np.ndarray | None
    pinch_src_xy_cm: float
    min_pinch_src_xy_cm: float
    src_z_cm: float
    max_src_z_cm: float
    src_tgt_xy_cm: float
    bystander_disp_cm: float
    task_success: bool
    status_text: str
    latency_ms: float


def project_3d_to_camera_pixels(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cam_name: str,
    points_3d: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D world coordinates (N, 3) into 2D pixel coordinates (N, 2) for a MuJoCo camera."""
    cam_id = model.camera(cam_name).id
    cam_pos = data.cam_xpos[cam_id]
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3)

    rel = points_3d - cam_pos[None, :]
    p_cam = rel @ cam_mat  # [:, 0]=right, [:, 1]=up, [:, 2]=back (-depth)

    x_c = p_cam[:, 0]
    y_c = p_cam[:, 1]
    z_c = p_cam[:, 2]

    depth = -z_c
    in_front = depth > 1e-3
    safe_depth = np.where(in_front, depth, 1.0)

    fovy_rad = np.radians(float(model.cam_fovy[cam_id]))
    focal_px = (0.5 * height) / np.tan(0.5 * fovy_rad)

    u = (0.5 * width) + focal_px * (x_c / safe_depth)
    v = (0.5 * height) - focal_px * (y_c / safe_depth)

    pixels_2d = np.stack([u, v], axis=-1).astype(np.float32)
    valid = (
        in_front
        & (u >= -12.0)
        & (u <= width + 12.0)
        & (v >= -12.0)
        & (v <= height + 12.0)
    )
    return pixels_2d, valid


class RolloutVisualizer:
    """Renders multi-camera diagnostic GIFs and 6-phase keyframe strips for eval rollouts."""

    CAM_COLORS: ClassVar[dict[str, tuple[int, int, int]]] = {
        "third_person_cam": (56, 189, 248),  # Sky Blue
        "overhead_cam": (52, 211, 153),      # Emerald
        "wrist_cam": (251, 113, 133),        # Rose
    }
    CAM_TITLES: ClassVar[dict[str, str]] = {
        "third_person_cam": "THIRD-PERSON CAM",
        "overhead_cam": "OVERHEAD CAM",
        "wrist_cam": "WRIST CAM",
    }

    def __init__(self, env: SimEnv, cam_size: int = 256) -> None:
        self.env = env
        self.cam_size = int(cam_size)
        self.viz_renderer = mujoco.Renderer(env.model, height=self.cam_size, width=self.cam_size)
        self.fk_data = mujoco.MjData(env.model)
        self.records: list[RolloutStepRecord] = []
        self.frames: list[Image.Image] = []

    def reset(self) -> None:
        """Clear recorded rollout frames before starting a new episode."""
        self.records.clear()
        self.frames.clear()

    def close(self) -> None:
        """Release the dedicated visualization renderer."""
        if self.viz_renderer is not None:
            self.viz_renderer.close()
            self.viz_renderer = None

    def compute_fk_pinch_trajectory(self, chunk_6d: np.ndarray) -> np.ndarray:
        """Compute 3D pinch_site positions (H + 1, 3) across an action chunk via Forward Kinematics."""
        horizon = chunk_6d.shape[0]
        pts_3d = np.zeros((horizon + 1, 3), dtype=np.float32)
        pts_3d[0] = self.env.data.site(self.env._pinch_site_id).xpos.copy().astype(np.float32)

        self.fk_data.qpos[:] = self.env.data.qpos[:]
        for h in range(horizon):
            for j_idx in range(5):
                q_adr = self.env._qpos_adrs[j_idx]
                self.fk_data.qpos[q_adr] = float(chunk_6d[h, j_idx])
            self.fk_data.qpos[self.env._qpos_adrs[5]] = norm_gripper_to_raw(float(chunk_6d[h, 5]))
            mujoco.mj_kinematics(self.env.model, self.fk_data)
            pts_3d[h + 1] = self.fk_data.site(self.env._pinch_site_id).xpos.copy().astype(np.float32)

        return pts_3d

    def record_step(
        self,
        *,
        step_idx: int,
        max_steps: int,
        task_id: int,
        episode_label: str,
        domain: str,
        pred_chunk: np.ndarray,
        gt_chunk: np.ndarray | None,
        keypoints: dict[str, np.ndarray],
        camera_weights: np.ndarray,
        proprio: np.ndarray,
        gt_proprio: np.ndarray | None,
        pinch_src_xy_cm: float,
        min_pinch_src_xy_cm: float,
        src_z_cm: float,
        max_src_z_cm: float,
        src_tgt_xy_cm: float,
        bystander_disp_cm: float,
        task_success: bool,
        status_text: str,
        latency_ms: float,
    ) -> None:
        """Capture and render one control step into a diagnostic dashboard frame."""
        spec = TASK_SPECS[task_id]
        src_pos_3d = self.env.data.body(self.env._obj_body_ids[spec.source_object]).xpos.copy()[None, :]
        tgt_pos_3d = self.env.data.body(self.env._obj_body_ids[spec.target_object]).xpos.copy()[None, :]

        pred_pts_3d = self.compute_fk_pinch_trajectory(pred_chunk)
        gt_pts_3d = self.compute_fk_pinch_trajectory(gt_chunk) if gt_chunk is not None else None

        cam_images: dict[str, np.ndarray] = {}
        pred_traj_2d: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        gt_traj_2d: dict[str, tuple[np.ndarray, np.ndarray]] | None = {} if gt_pts_3d is not None else None
        src_marker_2d: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        tgt_marker_2d: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        for cam in CAMERA_NAMES:
            self.viz_renderer.update_scene(self.env.data, camera=cam)
            cam_images[cam] = self.viz_renderer.render().copy()
            pred_traj_2d[cam] = project_3d_to_camera_pixels(
                self.env.model, self.env.data, cam, pred_pts_3d, self.cam_size, self.cam_size
            )
            if gt_pts_3d is not None and gt_traj_2d is not None:
                gt_traj_2d[cam] = project_3d_to_camera_pixels(
                    self.env.model, self.env.data, cam, gt_pts_3d, self.cam_size, self.cam_size
                )
            src_marker_2d[cam] = project_3d_to_camera_pixels(
                self.env.model, self.env.data, cam, src_pos_3d, self.cam_size, self.cam_size
            )
            tgt_marker_2d[cam] = project_3d_to_camera_pixels(
                self.env.model, self.env.data, cam, tgt_pos_3d, self.cam_size, self.cam_size
            )

        rec = RolloutStepRecord(
            step_idx=step_idx,
            max_steps=max_steps,
            task_id=task_id,
            episode_label=episode_label,
            domain=domain,
            cam_images=cam_images,
            keypoints=keypoints,
            camera_weights=camera_weights,
            pred_traj_2d=pred_traj_2d,
            gt_traj_2d=gt_traj_2d,
            src_marker_2d=src_marker_2d,
            tgt_marker_2d=tgt_marker_2d,
            proprio=proprio.copy(),
            gt_proprio=gt_proprio.copy() if gt_proprio is not None else None,
            pinch_src_xy_cm=pinch_src_xy_cm,
            min_pinch_src_xy_cm=min_pinch_src_xy_cm,
            src_z_cm=src_z_cm,
            max_src_z_cm=max_src_z_cm,
            src_tgt_xy_cm=src_tgt_xy_cm,
            bystander_disp_cm=bystander_disp_cm,
            task_success=task_success,
            status_text=status_text,
            latency_ms=latency_ms,
        )
        self.records.append(rec)
        self.frames.append(self._render_dashboard_frame(rec))

    def _render_dashboard_frame(self, rec: RolloutStepRecord) -> Image.Image:
        """Render a 3-camera + bottom HUD frame (768 x 432 px)."""
        cs = self.cam_size
        total_w = cs * 3
        hud_h = 176
        total_h = cs + hud_h

        canvas = Image.new("RGB", (total_w, total_h), (15, 23, 42))
        font = ImageFont.load_default()
        spec = TASK_SPECS[rec.task_id]

        # 1. Render each camera panel with overlays
        for idx, cam in enumerate(CAMERA_NAMES):
            panel = Image.fromarray(rec.cam_images[cam]).convert("RGBA")
            overlay = Image.new("RGBA", (cs, cs), (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            # Subtle 2D Spatial Softmax keypoints
            if cam in rec.keypoints:
                kp = rec.keypoints[cam]  # (K, 2) in [-1, 1]
                for k_idx in range(kp.shape[0]):
                    kx = (float(kp[k_idx, 0]) + 1.0) * 0.5 * (cs - 1)
                    ky = (float(kp[k_idx, 1]) + 1.0) * 0.5 * (cs - 1)
                    draw.ellipse(
                        (kx - 2.0, ky - 2.0, kx + 2.0, ky + 2.0),
                        fill=(253, 224, 71, 135),
                    )

            # Source and Target object rings on external cameras
            if cam in ("third_person_cam", "overhead_cam"):
                src_px, src_ok = rec.src_marker_2d[cam]
                if bool(src_ok[0]):
                    sx, sy = float(src_px[0, 0]), float(src_px[0, 1])
                    draw.ellipse((sx - 12, sy - 12, sx + 12, sy + 12), outline=(251, 191, 36, 230), width=2)
                    draw.text((sx - 10, sy - 24), "SRC", fill=(251, 191, 36, 255), font=font)

                tgt_px, tgt_ok = rec.tgt_marker_2d[cam]
                if bool(tgt_ok[0]):
                    tx, ty = float(tgt_px[0, 0]), float(tgt_px[0, 1])
                    draw.ellipse((tx - 14, ty - 14, tx + 14, ty + 14), outline=(52, 211, 153, 230), width=2)
                    draw.text((tx - 10, ty - 26), "TGT", fill=(52, 211, 153, 255), font=font)

            # Ground-truth expert 16-step trajectory ribbon (Green)
            if rec.gt_traj_2d is not None and cam in rec.gt_traj_2d:
                gt_px, gt_ok = rec.gt_traj_2d[cam]
                valid_gt = [
                    (float(gt_px[i, 0]), float(gt_px[i, 1]))
                    for i in range(gt_px.shape[0])
                    if bool(gt_ok[i])
                ]
                if len(valid_gt) >= 2:
                    draw.line(valid_gt, fill=(74, 222, 128, 210), width=3)
                    gx, gy = valid_gt[-1]
                    draw.ellipse((gx - 4, gy - 4, gx + 4, gy + 4), fill=(74, 222, 128, 255))

            # Policy predicted 16-step trajectory ribbon (Bright Cyan)
            pred_px, pred_ok = rec.pred_traj_2d[cam]
            valid_pred = [
                (float(pred_px[i, 0]), float(pred_px[i, 1]))
                for i in range(pred_px.shape[0])
                if bool(pred_ok[i])
            ]
            if len(valid_pred) >= 2:
                draw.line(valid_pred, fill=(56, 189, 248, 240), width=3)
                p0x, p0y = valid_pred[0]
                draw.ellipse((p0x - 3.5, p0y - 3.5, p0x + 3.5, p0y + 3.5), fill=(255, 255, 255, 255))
                p1x, p1y = valid_pred[-1]
                draw.ellipse((p1x - 4.5, p1y - 4.5, p1x + 4.5, p1y + 4.5), fill=(56, 189, 248, 255))

            # Top banner per camera
            draw.rectangle((0, 0, cs, 22), fill=(15, 23, 42, 195))
            cam_color = self.CAM_COLORS[cam]
            w_pct = float(rec.camera_weights[idx]) * 100.0
            draw.text((8, 5), f"{self.CAM_TITLES[cam]}  ({w_pct:.0f}%)", fill=(*cam_color, 255), font=font)

            merged = Image.alpha_composite(panel, overlay).convert("RGB")
            canvas.paste(merged, (idx * cs, 0))

        # 2. Draw Bottom Telemetry Dashboard
        hud = ImageDraw.Draw(canvas)
        y0 = cs
        hud.line((0, y0, total_w, y0), fill=(51, 65, 85), width=2)

        # Column 1: Task, Episode, Progress, Status Badge (x = 12..265)
        hud.text(
            (12, y0 + 10),
            f"TASK {rec.task_id}: {spec.name}",
            fill=(248, 250, 252),
            font=font,
        )
        hud.text(
            (12, y0 + 28),
            f"Goal: {spec.source_object} -> {spec.target_object}",
            fill=(148, 163, 184),
            font=font,
        )
        hud.text(
            (12, y0 + 46),
            f"Ep: {rec.episode_label}  |  Dom: {rec.domain}",
            fill=(148, 163, 184),
            font=font,
        )
        hud.text(
            (12, y0 + 64),
            f"Step: {rec.step_idx + 1:03d}/{rec.max_steps:03d}  ({rec.latency_ms:.1f} ms)",
            fill=(203, 213, 225),
            font=font,
        )

        # Progress bar
        bar_x0, bar_y0, bar_w, bar_h = 12, y0 + 84, 235, 10
        hud.rectangle((bar_x0, bar_y0, bar_x0 + bar_w, bar_y0 + bar_h), fill=(30, 41, 59), outline=(71, 85, 105))
        prog = min(1.0, float(rec.step_idx + 1) / float(max(rec.max_steps, 1)))
        hud.rectangle((bar_x0, bar_y0, bar_x0 + int(bar_w * prog), bar_y0 + bar_h), fill=(56, 189, 248))

        # Status badge
        badge_bg = (22, 101, 52) if rec.task_success else (127, 29, 29) if "FAIL" in rec.status_text else (30, 58, 138)
        badge_fg = (134, 239, 172) if rec.task_success else (252, 165, 165) if "FAIL" in rec.status_text else (147, 197, 253)
        hud.rectangle((12, y0 + 104, 247, y0 + 162), fill=badge_bg, outline=badge_fg)
        # Wrap status text over 2 lines if long
        s_line1 = rec.status_text[:34]
        s_line2 = rec.status_text[34:68]
        hud.text((18, y0 + 114), s_line1, fill=badge_fg, font=font)
        if s_line2:
            hud.text((18, y0 + 132), s_line2, fill=badge_fg, font=font)

        # Column 2: Physical Distances & Joint Tracking (x = 265..525)
        hud.line((258, y0 + 8, 258, total_h - 8), fill=(51, 65, 85), width=1)
        hud.text((268, y0 + 10), "PHYSICAL & JOINT TELEMETRY", fill=(248, 250, 252), font=font)
        hud.text(
            (268, y0 + 30),
            f"Pinch->Src XY: {rec.pinch_src_xy_cm:5.1f} cm (min {rec.min_pinch_src_xy_cm:4.1f} cm)",
            fill=(251, 191, 36),
            font=font,
        )
        hud.text(
            (268, y0 + 48),
            f"Source Height: {rec.src_z_cm:5.1f} cm (max {rec.max_src_z_cm:4.1f} cm)",
            fill=(56, 189, 248),
            font=font,
        )
        hud.text(
            (268, y0 + 66),
            f"Src->Tgt XY:   {rec.src_tgt_xy_cm:5.1f} cm | Byst: {rec.bystander_disp_cm:4.1f} cm",
            fill=(52, 211, 153),
            font=font,
        )
        grip_val = float(rec.proprio[5])
        hud.text(
            (268, y0 + 84),
            f"Gripper Jaw:   {grip_val:5.2f} ({'OPEN' if grip_val > 0.55 else 'CLAMPED'})",
            fill=(226, 232, 240),
            font=font,
        )

        pan_cur = float(rec.proprio[0])
        lift_cur = float(rec.proprio[1])
        if rec.gt_proprio is not None:
            pan_gt = float(rec.gt_proprio[0])
            lift_gt = float(rec.gt_proprio[1])
            pan_err = abs(pan_cur - pan_gt)
            pan_col = (248, 113, 113) if pan_err > 0.15 else (134, 239, 172)
            hud.text(
                (268, y0 + 106),
                f"Pan  q0: {pan_cur:+.2f} rad (GT {pan_gt:+.2f}, err {pan_err:.2f})",
                fill=pan_col,
                font=font,
            )
            hud.text(
                (268, y0 + 124),
                f"Lift q1: {lift_cur:+.2f} rad (GT {lift_gt:+.2f})",
                fill=(203, 213, 225),
                font=font,
            )
        else:
            hud.text(
                (268, y0 + 106),
                f"Pan  q0: {pan_cur:+.2f} rad | Lift q1: {lift_cur:+.2f} rad",
                fill=(203, 213, 225),
                font=font,
            )

        # Legend
        hud.text(
            (268, y0 + 146),
            "Legend: [Cyan]=Policy Chunk  [Green]=Expert GT",
            fill=(148, 163, 184),
            font=font,
        )

        # Column 3: Multi-Camera Attention Meters (x = 535..756)
        hud.line((528, y0 + 8, 528, total_h - 8), fill=(51, 65, 85), width=1)
        hud.text((538, y0 + 10), "CAMERA ATTENTION WEIGHTS", fill=(248, 250, 252), font=font)
        for c_idx, cam in enumerate(CAMERA_NAMES):
            cy = y0 + 34 + c_idx * 38
            w_val = float(rec.camera_weights[c_idx])
            c_col = self.CAM_COLORS[cam]
            short_name = cam.replace("_cam", "").upper()
            hud.text((538, cy), f"{short_name:12s} {w_val * 100:4.1f}%", fill=c_col, font=font)
            bx0, by0, bw, bh = 538, cy + 15, 210, 10
            hud.rectangle((bx0, by0, bx0 + bw, by0 + bh), fill=(30, 41, 59), outline=(71, 85, 105))
            hud.rectangle((bx0, by0, bx0 + int(bw * min(1.0, w_val)), by0 + bh), fill=c_col)

        hud.text(
            (538, y0 + 146),
            "[SRC]=Source Obj  [TGT]=Target Goal",
            fill=(148, 163, 184),
            font=font,
        )
        return canvas

    def save_gif(
        self,
        output_path: str | Path,
        fps: int = 15,
        frame_stride: int = 2,
    ) -> Path:
        """Save the recorded dashboard frames as an animated GIF."""
        if not self.frames:
            raise RuntimeError("No frames recorded in RolloutVisualizer.")
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        selected = self.frames[:: max(1, int(frame_stride))]
        if self.frames[-1] is not selected[-1]:
            selected.append(self.frames[-1])

        duration_ms = round(1000.0 / max(1, int(fps)))
        selected[0].save(
            out,
            save_all=True,
            append_images=selected[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
        )
        return out

    def save_summary_strip(
        self,
        output_path: str | Path,
        num_keyframes: int = 6,
    ) -> Path:
        """Save a 6-keyframe progression strip (.png) across the episode."""
        if not self.frames:
            raise RuntimeError("No frames recorded in RolloutVisualizer.")
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        n = len(self.frames)
        indices = np.linspace(0, n - 1, num=min(num_keyframes, n), dtype=int)
        keyframes = [self.frames[int(i)] for i in indices]

        fw, fh = keyframes[0].size
        # Stack in a 2x3 grid for easy reading in chat/markdown
        cols = 2
        rows = (len(keyframes) + cols - 1) // cols
        pad = 8
        grid = Image.new(
            "RGB",
            (cols * fw + (cols + 1) * pad, rows * fh + (rows + 1) * pad),
            (10, 15, 28),
        )
        for idx, frame in enumerate(keyframes):
            r = idx // cols
            c = idx % cols
            grid.paste(frame, (pad + c * (fw + pad), pad + r * (fh + pad)))

        grid.save(out)
        return out
