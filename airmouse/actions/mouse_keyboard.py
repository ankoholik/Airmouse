"""System mouse control via pyautogui and pynput."""

from __future__ import annotations

import time

import numpy as np
import pyautogui
from pynput.mouse import Button, Controller


class MouseKeyboardController:
    """Maps hand landmarks and gesture IDs to mouse actions."""

    GESTURE_MOVE_CURSOR = 2
    GESTURE_LEFT_CLICK = 3
    GESTURE_RIGHT_CLICK = 4
    GESTURE_SCROLL = 5

    def __init__(
        self,
        mouse_sensitivity: float = 1.0,
        camera_roi_margin: float = 0.08,
        scroll_step: int = 3,
        click_cooldown: float = 0.5,
        click_hold_threshold_s: float = 2.5,
    ) -> None:
        self.mouse_sensitivity = mouse_sensitivity
        self.camera_roi_margin = camera_roi_margin
        self.scroll_step = scroll_step
        self.click_cooldown = click_cooldown
        self.click_hold_threshold_s = click_hold_threshold_s

        self._mouse = Controller()

        self.cursor_x: float | None = None
        self.cursor_y: float | None = None
        self.last_gesture_time = 0.0
        self.last_gesture_id = 1
        self._prev_frame_gesture_id = 1
        self._ys: list[float] = []

        self._lmb_segment_start: float | None = None
        self._lmb_hold_engaged: bool = False
        self._rmb_segment_start: float | None = None
        self._rmb_hold_engaged: bool = False

    def reset(self) -> None:
        self.cursor_x = None
        self.cursor_y = None
        self.last_gesture_time = 0.0
        self.last_gesture_id = 1
        self._prev_frame_gesture_id = 1
        self._ys.clear()
        self._release_left_if_down()
        self._release_right_if_down()
        self._lmb_segment_start = None
        self._lmb_hold_engaged = False
        self._rmb_segment_start = None
        self._rmb_hold_engaged = False

    def _release_left_if_down(self) -> None:
        if self._lmb_hold_engaged:
            try:
                self._mouse.release(Button.left)
            finally:
                self._lmb_hold_engaged = False

    def _release_right_if_down(self) -> None:
        if self._rmb_hold_engaged:
            try:
                self._mouse.release(Button.right)
            finally:
                self._rmb_hold_engaged = False

    def move_cursor(self, hand_landmark, gesture_id: int, *, landmark_point: int | None = None) -> bool:
        screen_width, screen_height = pyautogui.size()

        if landmark_point is None:
            landmark_point = 8 if gesture_id != self.GESTURE_MOVE_CURSOR else 4
        x = hand_landmark.landmark[landmark_point].x
        y = hand_landmark.landmark[landmark_point].y

        m = float(self.camera_roi_margin)
        m = max(0.0, min(0.45, m))
        denom = max(1e-6, 1.0 - 2.0 * m)

        x_n = (float(x) - m) / denom
        y_n = (float(y) - m) / denom

        x_n = max(0.0, min(1.0, x_n))
        y_n = max(0.0, min(1.0, y_n))
        x_n = 1.0 - x_n

        if self.mouse_sensitivity != 1.0:
            s = float(self.mouse_sensitivity)
            x_n = 0.5 + (x_n - 0.5) * s
            y_n = 0.5 + (y_n - 0.5) * s
            x_n = max(0.0, min(1.0, x_n))
            y_n = max(0.0, min(1.0, y_n))

        target_x = 10.0 + x_n * float(screen_width - 20)
        target_y = 10.0 + y_n * float(screen_height - 20)

        alpha = 0.35
        if self.cursor_x is None or self.cursor_y is None:
            self.cursor_x, self.cursor_y = target_x, target_y
        else:
            self.cursor_x = (1 - alpha) * self.cursor_x + alpha * target_x
            self.cursor_y = (1 - alpha) * self.cursor_y + alpha * target_y

        self._mouse.position = (int(self.cursor_x), int(self.cursor_y))
        self.last_gesture_id = gesture_id
        return True

    def _finalize_left_segment(self) -> None:
        if self._lmb_segment_start is None:
            return
        if self._lmb_hold_engaged:
            try:
                self._mouse.release(Button.left)
            finally:
                self._lmb_hold_engaged = False
        else:
            if time.time() - self.last_gesture_time >= self.click_cooldown:
                self._mouse.click(Button.left, 1)
                self.last_gesture_time = time.time()
        self._lmb_segment_start = None

    def _finalize_right_segment(self) -> None:
        if self._rmb_segment_start is None:
            return
        if self._rmb_hold_engaged:
            try:
                self._mouse.release(Button.right)
            finally:
                self._rmb_hold_engaged = False
        else:
            if time.time() - self.last_gesture_time >= self.click_cooldown:
                self._mouse.click(Button.right, 1)
                self.last_gesture_time = time.time()
        self._rmb_segment_start = None

    def _on_gesture_changed_from_prev(self, gesture_id: int) -> None:
        if self._prev_frame_gesture_id == self.GESTURE_LEFT_CLICK and gesture_id != self.GESTURE_LEFT_CLICK:
            self._finalize_left_segment()
        if self._prev_frame_gesture_id == self.GESTURE_RIGHT_CLICK and gesture_id != self.GESTURE_RIGHT_CLICK:
            self._finalize_right_segment()

    def handle_gesture(self, hand_landmark, gesture_id: int) -> bool:
        # Track wrist Y once per frame so scroll direction depends on
        # actual hand (wrist) up/down movement even if we don't move the cursor.
        try:
            wrist_y = float(hand_landmark.landmark[0].y)
        except Exception:
            wrist_y = None
        if wrist_y is not None:
            self._ys.append(wrist_y)
            # Prevent unbounded growth on long sessions.
            if len(self._ys) > 30:
                del self._ys[:10]

        self._on_gesture_changed_from_prev(gesture_id)

        if gesture_id == self.GESTURE_MOVE_CURSOR:
            ok = self.move_cursor(hand_landmark, gesture_id)
            self._prev_frame_gesture_id = gesture_id
            return ok

        if gesture_id == self.GESTURE_LEFT_CLICK:
            # Do not move cursor while "click" gesture is active.
            # Otherwise switching between different landmarks (move vs click)
            # causes a visible cursor jump and makes clicking inaccurate.
            now = time.time()
            if self._lmb_segment_start is None:
                self._lmb_segment_start = now
            if (
                not self._lmb_hold_engaged
                and (now - self._lmb_segment_start) >= self.click_hold_threshold_s
            ):
                self._mouse.press(Button.left)
                self._lmb_hold_engaged = True
            # While holding (drag/select), allow moving the cursor again,
            # but use the same landmark as MOVE to avoid jumps.
            if self._lmb_hold_engaged:
                self.move_cursor(hand_landmark, self.GESTURE_MOVE_CURSOR, landmark_point=4)
            self._prev_frame_gesture_id = gesture_id
            return True

        if gesture_id == self.GESTURE_RIGHT_CLICK:
            # Keep cursor fixed during right-click gesture for accuracy.
            now = time.time()
            if self._rmb_segment_start is None:
                self._rmb_segment_start = now
            if (
                not self._rmb_hold_engaged
                and (now - self._rmb_segment_start) >= self.click_hold_threshold_s
            ):
                self._mouse.press(Button.right)
                self._rmb_hold_engaged = True
            if self._rmb_hold_engaged:
                self.move_cursor(hand_landmark, self.GESTURE_MOVE_CURSOR, landmark_point=4)
            self._prev_frame_gesture_id = gesture_id
            return True

        if gesture_id == self.GESTURE_SCROLL:
            # Scrolling should not drag the cursor around.
            if len(self._ys) >= 6:
                # MediaPipe normalized Y grows downward:
                # - hand up   => Y decreases
                # - hand down => Y increases
                curr = float(np.mean(self._ys[-3:]))
                prev = float(np.mean(self._ys[-6:-3]))
                dy = curr - prev

                # Small threshold to avoid jitter-based micro scrolls.
                eps = 0.004
                if dy <= -eps:
                    # Hand moved up => scroll up
                    self._mouse.scroll(0, self.scroll_step)
                elif dy >= eps:
                    # Hand moved down => scroll down
                    self._mouse.scroll(0, -self.scroll_step)
                self._prev_frame_gesture_id = gesture_id
                return True
            self._prev_frame_gesture_id = gesture_id
            return False

        self._prev_frame_gesture_id = gesture_id
        return False
