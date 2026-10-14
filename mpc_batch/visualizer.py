"""
Highway Trajectory and Multi-Modal MPC Visualizer.
Renders high-quality animations and static snapshots of the highway driving scenario,
matching Figure 1 and `highway_car2.py` from the paper.
"""

from typing import List, Tuple, Dict, Any, Optional
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import matplotlib.animation as animation

from .controller import StepTelemetry, EgoState


class HighwayVisualizer:
    """
    Renders highway environment, surrounding traffic, ego vehicle,
    and all batched candidate trajectories with the winning trajectory highlighted.
    """
    def __init__(
        self,
        lanes: List[float] = [-10.0, -6.0, -2.0, 2.0, 6.0, 10.0],
        a_ell: float = 5.6,
        b_ell: float = 3.0,
        a_rect: float = 4.0,
        b_rect: float = 1.4,
        window_ahead: float = 140.0,
        window_behind: float = 30.0,
        figsize: Tuple[int, int] = (16, 7),
    ):
        self.lanes = lanes
        self.a_ell = a_ell
        self.b_ell = b_ell
        self.a_rect = a_rect
        self.b_rect = b_rect
        self.window_ahead = window_ahead
        self.window_behind = window_behind
        self.figsize = figsize
        
        self.diag = math.sqrt(a_rect ** 2 + b_rect ** 2)
        self.rect_angle_offset = math.atan2(b_rect, a_rect)

    def draw_frame(
        self,
        ax: plt.Axes,
        telemetry: StepTelemetry,
        all_vehicles: Optional[List[Any]] = None,
    ):
        """Draws a single highway scene frame onto the given matplotlib Axes."""
        ax.clear()
        
        ego = telemetry.ego_state
        res = telemetry.result
        best_idx = telemetry.best_goal_idx
        
        lower_x = ego.x - self.window_behind
        upper_x = ego.x + self.window_ahead
        
        # 1. Draw highway boundaries and lane dividers
        ax.axhline(12.0, color="black", linestyle="-", linewidth=2.5, alpha=0.6)
        ax.axhline(-12.0, color="black", linestyle="-", linewidth=2.5, alpha=0.6)
        
        # Dashed dividers between lanes
        lane_dividers = [-8.0, -4.0, 0.0, 4.0, 8.0]
        for div_y in lane_dividers:
            ax.axhline(div_y, color="gray", linestyle="--", linewidth=1.2, alpha=0.5)
            
        # Draw lane centerlines faintly
        for lane_y in self.lanes:
            ax.axhline(lane_y, color="lightgray", linestyle=":", linewidth=0.8, alpha=0.4)
            
        # 2. Draw all candidate trajectories
        x_all = res.x.detach().cpu().numpy() # [L, num]
        y_all = res.y.detach().cpu().numpy() # [L, num]
        L = x_all.shape[0]
        
        # Colors for candidate trajectories (distinguishable palette)
        try:
            cmap = plt.get_cmap("tab10")
        except AttributeError:
            cmap = plt.cm.tab10
        for i in range(L):
            if i != best_idx:
                ax.plot(
                    x_all[i, 2:],
                    y_all[i, 2:],
                    color=cmap(i % 10),
                    linewidth=1.5,
                    alpha=0.45,
                    linestyle="-",
                )
                ax.scatter(x_all[i, -1], y_all[i, -1], color=cmap(i % 10), s=20, alpha=0.5)
                
        # 3. Draw winning / selected optimal trajectory in bold red
        ax.plot(
            x_all[best_idx, 2:],
            y_all[best_idx, 2:],
            color="red",
            linewidth=3.5,
            alpha=1.0,
            label="Optimal Trajectory (Lowest Meta-Cost)",
            zorder=15,
        )
        ax.scatter(
            x_all[best_idx, -1],
            y_all[best_idx, -1],
            color="red",
            s=80,
            marker="*",
            edgecolor="darkred",
            zorder=16,
        )
        
        # 4. Draw surrounding obstacle vehicles
        if all_vehicles is not None:
            for veh in all_vehicles:
                if lower_x - 10 <= veh.x <= upper_x + 10:
                    # Obstacle ellipse
                    ell = Ellipse(
                        xy=(veh.x, veh.y),
                        width=self.a_ell,
                        height=self.b_ell,
                        angle=0.0,
                        facecolor="#38b6ff",
                        edgecolor="#004aad",
                        alpha=0.35,
                        zorder=5,
                    )
                    ax.add_patch(ell)
                    
                    # Obstacle rectangular body
                    rect_corner_x = veh.x - (self.diag / 2.0) * math.cos(veh.psi + self.rect_angle_offset)
                    rect_corner_y = veh.y - (self.diag / 2.0) * math.sin(veh.psi + self.rect_angle_offset)
                    rect = Rectangle(
                        xy=(rect_corner_x, rect_corner_y),
                        width=self.a_rect,
                        height=self.b_rect,
                        angle=math.degrees(veh.psi),
                        facecolor="#0052cc",
                        edgecolor="black",
                        linewidth=1.2,
                        alpha=0.85,
                        zorder=6,
                    )
                    ax.add_patch(rect)
                    
                    # Velocity label
                    ax.text(
                        veh.x - 1.5,
                        veh.y - 0.25,
                        f"{veh.vx:.1f}",
                        color="white",
                        fontsize=8,
                        fontweight="bold",
                        zorder=7,
                    )
                    
        # 5. Draw ego vehicle (Red)
        ego_ell = Ellipse(
            xy=(ego.x, ego.y),
            width=self.a_ell,
            height=self.b_ell,
            angle=0.0,
            facecolor="#ff7b7b",
            edgecolor="#cc0000",
            alpha=0.4,
            zorder=10,
        )
        ax.add_patch(ego_ell)
        
        ego_rect_corner_x = ego.x - (self.diag / 2.0) * math.cos(ego.psi + self.rect_angle_offset)
        ego_rect_corner_y = ego.y - (self.diag / 2.0) * math.sin(ego.psi + self.rect_angle_offset)
        ego_rect = Rectangle(
            xy=(ego_rect_corner_x, ego_rect_corner_y),
            width=self.a_rect,
            height=self.b_rect,
            angle=math.degrees(ego.psi),
            facecolor="#e60000",
            edgecolor="black",
            linewidth=1.5,
            alpha=0.9,
            zorder=11,
        )
        ax.add_patch(ego_rect)
        
        # Ego velocity label
        ax.text(
            ego.x - 1.5,
            ego.y - 0.25,
            f"{ego.v:.1f}",
            color="white",
            fontsize=8,
            fontweight="bold",
            zorder=12,
        )
        
        # 6. Axis limits and labels
        ax.set_xlim(lower_x, upper_x)
        ax.set_ylim(-13.0, 16.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Longitudinal Distance X (m)", fontsize=11)
        ax.set_ylabel("Lateral Position Y (m)", fontsize=11)
        
        # 7. Information Dashboard Header
        collision_str = "COLLISION DETECTED!" if telemetry.collision else "Safe"
        collision_color = "red" if telemetry.collision else "green"
        
        dashboard_text = (
            f"Step: {telemetry.step} | Time: {telemetry.sim_time:.2f}s | "
            f"Ego Speed: {ego.v:.2f} m/s | Heading: {math.degrees(ego.psi):.1f}° | "
            f"Selected Goal: #{best_idx+1}/{L} | Solved in: {telemetry.computation_time*1000:.1f}ms | "
            f"Status: {collision_str}"
        )
        ax.set_title(dashboard_text, fontsize=11, fontweight="bold", pad=12)

    def save_snapshot(
        self,
        telemetry: StepTelemetry,
        all_vehicles: List[Any],
        filepath: str,
    ):
        """Renders and saves a single snapshot of the current MPC state."""
        fig, ax = plt.subplots(figsize=self.figsize)
        self.draw_frame(ax, telemetry, all_vehicles)
        plt.tight_layout()
        plt.savefig(filepath, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def create_animation(
        self,
        telemetry_history: List[StepTelemetry],
        vehicle_states_history: List[List[Any]],
        output_filepath: str,
        fps: int = 12,
    ):
        """
        Creates and saves a video or animated GIF of the full driving scenario.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        def update(frame_idx):
            telem = telemetry_history[frame_idx]
            vehs = vehicle_states_history[frame_idx] if frame_idx < len(vehicle_states_history) else None
            self.draw_frame(ax, telem, vehs)
            return ax,
            
        anim = animation.FuncAnimation(
            fig,
            update,
            frames=len(telemetry_history),
            interval=1000.0 / fps,
            blit=False,
        )
        
        plt.tight_layout()
        if output_filepath.endswith(".gif"):
            anim.save(output_filepath, writer="pillow", fps=fps)
        else:
            anim.save(output_filepath, writer="ffmpeg", fps=fps)
        plt.close(fig)
