"""Compatibility composition for box-support responsibilities."""

from . import box_carry as _box_carry
from . import box_execution as _box_execution
from . import box_force_clamp as _box_force_clamp
from . import box_geometry as _box_geometry
from . import box_perception as _box_perception
from . import box_preparation as _box_preparation
from .box_carry import BoxCarryMixin
from .box_execution import BoxExecutionMixin
from .box_force_clamp import BoxForceClampMixin
from .box_geometry import BoxGeometryMixin
from .box_perception import BoxPerceptionMixin
from .box_preparation import BoxPreparationMixin


class BoxSupportMixin(
    BoxGeometryMixin,
    BoxPreparationMixin,
    BoxCarryMixin,
    BoxExecutionMixin,
    BoxForceClampMixin,
    BoxPerceptionMixin,
):
    """Preserve the existing mixin API while delegating by responsibility."""

    pass


# Legacy method bodies use BoxSupportMixin for explicit static dispatch.
# Bind that symbol in each extracted module to this composed facade.
for _module in (
    _box_geometry,
    _box_preparation,
    _box_carry,
    _box_execution,
    _box_force_clamp,
    _box_perception,
):
    _module.BoxSupportMixin = BoxSupportMixin
del _module
