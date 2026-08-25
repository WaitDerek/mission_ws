# RealBot S2 description

`urdf/realbots29.urdf` keeps the original exported link and joint names, while
both arm-end camera-support visuals use the replacement geometry derived from
the 305G + 90D assembly:

- CAD source: `../305G+90D相机支架装配.STEP`
- URDF mesh: `meshes/305G_90D_camera_bracket_assembly.STL`
- Links using it: `left_camera_Link`, `right_camera_Link`

The STEP assembly contains both an `arm_8` reference body and the complete 90D
gripper reference.  The arm reference is used to recover the mounting transform;
both reference groups are excluded from the generated STL.  The retained mesh
contains the 305G camera, camera supports, adapter flange, and mounting hardware,
so neither the existing arm meshes nor the gripper are duplicated.
The recovered CAD-to-`arm_8` mapping, in metres, is:

```text
arm_x =  cad_y - 0.09533512964844704
arm_y = -cad_x - 0.42726799845695496
arm_z =  cad_z - 1.0911436983381027
```

The final STL is additionally expressed in the existing `camera_Link` frame by
applying the inverse of `right_camera_joint`; the left joint differs by less
than one micrometre, so one shared mesh is used for both sides.

The retained assembly already contains the 305G camera geometry; therefore the
legacy camera child links remain as fixed coordinate/inertia frames but no
longer carry their old visual or collision meshes.

The original left/right camera STL files are retained for recovery but are no
longer referenced by the URDF.

The original left and right gripper Link/Joint subtrees were removed because
they occupy the same arm-end mounting region as the replacement camera
assembly.  Their STL files are retained in `meshes/` for recovery but are not
referenced by the current URDF.
