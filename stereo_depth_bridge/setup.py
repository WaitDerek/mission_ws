from glob import glob
from setuptools import find_packages, setup


package_name = "stereo_depth_bridge"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [f"resource/{package_name}"],
        ),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ZENG-Pika",
    maintainer_email="281288719@qq.com",
    description="Prepare calibrated stereo RGB for Isaac ROS ESS depth inference.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "stereo_rectifier = stereo_depth_bridge.stereo_rectifier:main",
        ]
    },
)
