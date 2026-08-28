from glob import glob
from pathlib import Path
from setuptools import find_packages, setup


package_name = "mission_controller"


def _install_tree(root: str, pattern: str) -> list[tuple[str, list[str]]]:
    grouped: dict[str, list[str]] = {}
    for filename in glob(f"{root}/**/{pattern}", recursive=True):
        relative_parent = Path(filename).parent
        destination = Path("share") / package_name / relative_parent
        grouped.setdefault(str(destination), []).append(filename)
    return [(destination, sorted(files)) for destination, files in grouped.items()]


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        *_install_tree("config", "*.yaml"),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools", "paho-mqtt>=1.5,<3"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="dekc",
    maintainer_email="dekc@example.com",
    description="RealBot box manipulation and depalletizing workflow orchestration.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "mission_controller = mission_runtime.mission_controller:main",
            "execute_workflow = mission_runtime.taskflow.node:main",
            "realbots_global_tf = mission_runtime.global_tf_publisher:main",
        ],
    },
)
