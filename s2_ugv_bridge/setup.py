from setuptools import find_packages, setup


package_name = "s2_ugv_bridge"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="dekc",
    maintainer_email="dekc@example.com",
    description="ROS2 host command gateway for the S2 ROS1 timed translation Action.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "timed_translate = s2_ugv_bridge.cli:main",
            "move_base_distance = s2_ugv_bridge.action_server:main",
        ],
    },
)
