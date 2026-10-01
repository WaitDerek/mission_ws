# 弯腰后的放置下放：固定实测起点，共同累计位移（2026-09-27）

本次仅修改动态 `/place_box_test` 弯腰完成后、触底前的分段下放。

1. 腰部到位后，等待两臂新鲜关节位置/速度反馈稳定，再采集稳定力基线。不发送手臂准备运动、不清零传感器。
2. 一次性保存左右臂实际 Link8 TF 和对应 arm base 变换，仅在本次 Action 下放中使用。两侧原有高度差保留，不取平均。
3. 两侧共用累计下降量，直接沿各自 arm base / SDK Work 的 -Z 生成普通 MoveL 目标。X和姿态保持实测起点值；每段正常下放左Work Y增加2mm、右Work Y减少2mm，使用从固定起点累计的绝对偏移，不从噪声反馈累加。不搜索 Y、不进行下降前的中点/终点离线 IK 筛选，也不检查负 Joint4、2°余量或10°分支变化。控制器仍执行其自身的 MoveL 规划与保护。
4. 转到左右臂工作坐标系，使用现有双线程阻塞 SDK MoveL；两侧完成后才能开始下一段。这不是控制器级同步。
5. 每段结束等待完成时刻之后的新鲜TF，在同一TF时刻读取左右TCP，核对各自Work-Z目标误差及左右Z差相对起点的变化。通过后记录共同指令下降量、左右实际累计下降量、差值、实际 XYZ 和整段耗时。这是段后验证，不是控制器级同步或运动中监督。
6. 保留原有段后 SDK 读力和松夹前承重确认。不新增运动中测力回调，不加入受力转移/桌面承重分类算法。

## 直接 MoveL，下放前不做 Y/IK 搜索

历史 `place_box_test_descent_y_search_*`、`place_box_test_descent_y_max_step_m` 和 `place_box_test_descent_ik_max_joint_step_deg` 参数为兼容旧配置保留声明，但本下放路径不再读取；无论旧开关是什么值，都不运行搜索。

普通 MoveL 必须指定完整 Pose，不能把 Y 写成“任意值”让控制器自动寻找路径。按用户要求指定每段Y位移：`place_box_test_descent_work_y_step_m=0.002`，左+、右−；累计上限 `place_box_test_descent_work_y_max_travel_m=0.100`。到上限后只下放Z，不为达到10cm额外运动。首次触底信号出现后冻结Y，不在接触桌面后继续横移；已有触底后3cm动作和单侧Z补偿不附加Y位移。若控制器仍报不可达，不自动绕行或重试。

已删除 Work-Y 平行性检查。下降方向按各自 arm base -Z 计算，而不是硬编码共同 base_link -Z；本方案依赖用户选定的放置腰角使两侧Z方向一致。更改放置腰角后应重新确认此几何条件，否则两臂等量Work-Z运动不一定保持箱体刚性关系。

不从平均料箱 Pose 和历史抓取关系重建末端目标；料箱估计仅用于已有的有限桌面搜索范围。每段前用各自Work-Z计算两侧相对原始起点的实测累计下降量，下一共同目标为 `min(固定总下降上限, max(上次指令累计量, 左实测量, 右实测量) + 本段步长)`。例如上次指令15mm、右臂实测20.789mm，下一步5mm生成累计25.789mm目标，不再回拉到20mm。起始TCP、初始高度差、XY/姿态参考和总距离预算不重置；只更新共同进度标量。剩余预算不足则缩短步长，预算已耗尽则不发新运动、不释放；控制器错误不自动重发。调用循环也更新累计指令量，避免下一次仍使用旧15/20mm进度。段后Z容差检查和有界单侧补偿保留，超过允许超调范围仍停止。

保留 SDK 返回码、取消、超时、有限下降范围及触底力判断。段后Z检查新增显式TF时间戳新鲜度检查；其余最新TF查询本身不等同新鲜度保证。未改动腰部单独弯腰、释放和回零流程，也未修改任何速度或力阈值配置。原有触底后额外3cm动作仍存在，其每段也增加下述Z检查。

## 每段Work-Z检查和向下补偿

比较各自 `L_base_Link/R_base_Link` 的Z，而非直接把共同坐标系Z当作Work-Z：

1. `e_left/right = actual_Work_Z - segment_target_Work_Z`。
2. `difference_change = (actual_left_Z - actual_right_Z) - (start_left_Z - start_right_Z)`。

两项默认容差均5mm；保留初始合理高度差，不强制左右Z相等。每次确认要求连续3个不同时间戳样本（复用 `place_box_test_stable_samples`），TF年龄≤0.25s。等待1s期间不发运动。

确认某侧未降够且读数稳定后，选正向目标误差最大的一侧，沿其Work -Z执行一次普通 `execute_single(..., motion_mode="movel")`；其X/Y与姿态保持当时实际值，另一臂不下发新命令。补偿后重新读取新鲜TF并确认两项误差。原始本段目标及整个下放起点均不改变。

默认参数：

```yaml
place_box_test_descent_z_target_tolerance_m: 0.005
place_box_test_descent_z_difference_tolerance_m: 0.005
place_box_test_descent_z_check_timeout_sec: 1.0
place_box_test_descent_z_feedback_max_age_sec: 0.25
place_box_test_descent_z_correction_max_attempts: 3
place_box_test_descent_z_correction_step_m: 0.002
place_box_test_descent_z_correction_max_travel_m: 0.005
```

每段最多3次补偿（两臂合计），单次≤2mm，单侧累计指令补偿≤5mm，且不越过原目标。补偿速度沿用下放速度百分比，每次SDK等待最多5s。SDK报错不重发；TF过期、仍在明显变化、补偿后不收敛、需要向上修正，或某侧已超调超限时停止双臂，不进入下一段或自动释放。不会无界跟随另一侧的超调位置继续下降。

新反馈：`PLACE_DESCENT_Z_CORRECTING`、`PLACE_DESCENT_Z_VERIFIED`、`PLACE_DESCENT_Z_CHECK_FAILED`，包含两侧实际/目标Z、初始Z差、Z差变化、次数及累计补偿距离。没有增加运动中测力回调或受力转移分类。

测试覆盖用户日志中的5.789mm超调、连续多段超调进度更新、总预算截断/耗尽、XY/姿态漂移不重设起点、保持初始高度差、倾斜 arm base 坐标转换、拒绝无效距离、超调后禁止上抬、SDK 失败及关节静止检查。测试在独立 ROS 域 197、localhost 中执行，未运行实机放置。

完成常规 merge-install 构建后需手动重启 Mission。新日志为 `PLACE_DESCENT_REFERENCE` 和逐段 `PLACE_DESCENT_PROGRESS`，其中 `origin=fixed_after_waist` 表示固定起点逻辑。`PLACE_DESCENT_PROGRESS_REBASED`记录名义目标、实测进度、更新后的共同目标和固定距离上限；REBASED只指进度调整，不代表重新设定TCP起点。未修改5mm默认Z容差，运行时设置20mm的用户应在重启后按需恢复该参数。
