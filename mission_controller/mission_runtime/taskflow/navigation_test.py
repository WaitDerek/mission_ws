"""Navigation-only workflow Action: no perception or manipulation clients."""
import time

from mission_interfaces.action import ExecuteNavigationTest
from rclpy.action import CancelResponse, GoalResponse

from .identifiers import new_workflow_id
from .model import NavigationRequest
from .mqtt_navigation import parse_navigation_points_json


class NavigationTestMixin:
    def _navigation_test_goal_callback(self, request):
        if not request.start:
            return GoalResponse.REJECT
        try:
            points = parse_navigation_points_json(self._string('mqtt_navigation_points_json'))
            if any(str(i) not in points for i in range(1, 8)):
                raise ValueError('navigation test requires configured points 1..7')
        except ValueError as exc:
            self.get_logger().warning(str(exc))
            return GoalResponse.REJECT
        with self._workflow_lock:
            if self._workflow_reserved:
                return GoalResponse.REJECT
            # Also exclude standalone NavigateToPoint and micro-navigation.
            response = self._reserve_goal('navigation_test', request.request_id)
            if response != GoalResponse.ACCEPT:
                return response
            self._workflow_reserved = True
            self._cancel_event.clear()
        return GoalResponse.ACCEPT

    def _navigation_test_cancel_callback(self, _goal_handle):
        self._cancel_event.set()
        gateway = self._active_navigation_gateway
        if gateway is not None:
            gateway.cancel_active()
        return CancelResponse.ACCEPT

    def _execute_navigation_test(self, goal_handle):
        request = goal_handle.request
        result = ExecuteNavigationTest.Result()
        result.workflow_id = new_workflow_id()
        result.final_stage = 'VALIDATING'
        rounds = int(request.rounds) or 1
        started = time.monotonic()
        gateway = None
        lease_token = ''
        current_round, current_point = 0, 0
        current_pos = []

        def canceled():
            return self._cancel_event.is_set() or bool(goal_handle.is_cancel_requested)

        def feedback(stage, detail):
            result.final_stage = stage
            msg = ExecuteNavigationTest.Feedback()
            msg.workflow_id = result.workflow_id
            msg.stage = stage
            msg.current_round = current_round
            msg.total_rounds = rounds
            msg.point_id = current_point
            msg.target_pos = current_pos
            msg.completed_navigation_count = result.completed_navigation_count
            msg.elapsed_sec = time.monotonic() - started
            msg.detail = detail
            goal_handle.publish_feedback(msg)
            self._audit_feedback(goal_handle, msg)

        with self._workflow_lock:
            self._active_goal_handle = goal_handle
        try:
            points = parse_navigation_points_json(self._string('mqtt_navigation_points_json'))
            # Freeze all seven poses before any request is sent.
            route = [(i, points[str(i)]) for i in range(1, 8)]
            feedback('VALIDATING', 'route 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7; navigation only')
            if canceled():
                raise InterruptedError('navigation test canceled before start')
            if not request.dry_run:
                feedback('ACQUIRE_LEASE', 'acquiring exclusive Mission ownership')
                acquired = self._acquire_lease(result.workflow_id)
                if not acquired.success:
                    raise RuntimeError(acquired.message)
                lease_token = acquired.lease_token
                if canceled():
                    raise InterruptedError('navigation test canceled before connection')
                feedback('CONNECTING', 'connecting to existing navigation gateway')
                gateway = self._make_navigation_gateway()
                self._active_navigation_gateway = gateway
            for current_round in range(1, rounds + 1):
                for current_point, target in route:
                    if canceled():
                        raise InterruptedError('navigation test canceled')
                    current_pos = [target.x, target.y, target.yaw]
                    feedback('DRY_RUN' if request.dry_run else 'NAVIGATING',
                             f'round {current_round}/{rounds}, point {current_point}, pos={current_pos}')
                    leg_start = time.monotonic()
                    result.attempted_point_ids.append(current_point)
                    try:
                        if not request.dry_run:
                            navigation = gateway.navigate(
                                NavigationRequest(
                                    workflow_id=result.workflow_id,
                                    step_id=f'{result.workflow_id}:round-{current_round}:point-{current_point}',
                                    point_id=str(current_point), pos=tuple(current_pos)),
                                canceled)
                            if canceled() or navigation.status == 'canceled':
                                raise InterruptedError(navigation.message or 'navigation canceled')
                            if not navigation.success:
                                raise RuntimeError(f'point {current_point}: {navigation.status}; {navigation.message}')
                    finally:
                        result.leg_elapsed_sec.append(time.monotonic() - leg_start)
                    result.completed_navigation_count += 1
                    feedback('POINT_COMPLETE', f'point {current_point} completed')
                result.completed_rounds += 1
            if canceled():
                raise InterruptedError('navigation test canceled')
            result.success = True
            result.message = (('dry-run validated ' if request.dry_run else 'completed ') +
                              f'{rounds} round(s), {result.completed_navigation_count} navigation legs')
            result.final_stage = 'DRY_RUN_COMPLETE' if request.dry_run else 'SUCCEEDED'
        except InterruptedError as exc:
            result.final_stage = 'CANCELED'
            result.failed_point_id = current_point
            result.message = str(exc)
        except Exception as exc:
            result.failed_point_id = current_point
            result.message = f'{result.final_stage}: {exc}'
            result.final_stage = 'FAILED'
        finally:
            cleanup_errors = []
            self._active_navigation_gateway = None
            if gateway is not None:
                try:
                    if not result.success:
                        gateway.cancel_active()
                    gateway.close()
                except Exception as exc:
                    cleanup_errors.append(f'navigation cleanup: {exc}')
            if lease_token:
                try:
                    released = self._release_lease(result.workflow_id, lease_token)
                    if not released.success:
                        cleanup_errors.append(f'lease release: {released.message}')
                except Exception as exc:
                    cleanup_errors.append(f'lease release: {exc}')
            self.mission_lease_manager.release_goal()
            with self._workflow_lock:
                self._workflow_reserved = False
                self._active_goal_handle = None
            if cleanup_errors:
                result.success = False
                result.message += '; ' + '; '.join(cleanup_errors)
                if result.final_stage != 'CANCELED':
                    result.final_stage = 'CLEANUP_FAILED'
        result.total_elapsed_sec = time.monotonic() - started
        if canceled():
            result.success = False
            result.final_stage = 'CANCELED'
        feedback(result.final_stage, result.message)
        if result.final_stage == 'CANCELED':
            goal_handle.canceled()
        elif result.success:
            goal_handle.succeed()
        else:
            goal_handle.abort()
        return result
