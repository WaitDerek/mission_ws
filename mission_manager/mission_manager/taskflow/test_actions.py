"""Independent single-point and route navigation Action endpoints."""

import math
import uuid

from mission_manager_interfaces.action import NavigateToPoint, ExecuteNavigation
from rclpy.action import ActionServer, GoalResponse

from .model import NavigationRequest
from .operations import RosWorkflowOperations


class TestActionsMixin:
    def _initialize_test_actions(self):
        self._navigation_server = ActionServer(
            self, NavigateToPoint, self._string("navigate_to_point_action_name"),
            execute_callback=self._execute_navigation,
            goal_callback=self._navigation_goal, cancel_callback=self._cancel_callback,
            callback_group=self._server_group)
        self._route_server = ActionServer(
            self, ExecuteNavigation,
            self._string("navigation_action_name"),
            execute_callback=self._execute_navigation_route,
            goal_callback=self._goal_callback, cancel_callback=self._cancel_callback,
            callback_group=self._server_group)

    def _navigation_goal(self, request):
        if request.point_id not in range(1, 5):
            return GoalResponse.REJECT
        if request.use_custom_pos and (
            len(request.pos) != 3 or not all(math.isfinite(x) for x in request.pos)
        ):
            return GoalResponse.REJECT
        return self._goal_callback(request)

    def _execute_navigation_route(self, goal_handle):
        return self._execute_flow(goal_handle, ExecuteNavigation, RosWorkflowOperations)

    def _execute_navigation(self, goal_handle):
        result = NavigateToPoint.Result()
        request = goal_handle.request
        result.point_id = request.point_id
        try:
            target = self._navigation_gateway._point_poses.get(str(request.point_id))
            if request.use_custom_pos:
                result.pos = list(request.pos)
            elif target is not None:
                result.pos = [target.x, target.y, target.yaw]
            else:
                raise ValueError("navigation point has no configured coordinates")
            feedback = NavigateToPoint.Feedback()
            feedback.stage = "NAVIGATING"
            feedback.detail = f"point {request.point_id}, pos={list(result.pos)}"
            goal_handle.publish_feedback(feedback)
            if self._cancel_event.is_set():
                result.message = "navigation canceled"
            elif request.dry_run:
                result.success = True
                result.message = "dry run: target validated; navigation not sent"
            else:
                outcome = self._navigation_gateway.navigate(
                    NavigationRequest(uuid.uuid4().hex, request.request_id,
                                      str(request.point_id), tuple(result.pos)),
                    self._cancel_event.is_set)
                result.success = outcome.success
                result.message = outcome.message
        except Exception as exc:
            result.message = f"navigation failed: {exc}"
        try:
            if self._cancel_event.is_set():
                result.success = False
                goal_handle.canceled()
            elif result.success:
                goal_handle.succeed()
            else:
                goal_handle.abort()
        finally:
            with self._workflow_lock:
                self._workflow_reserved = False
        return result
