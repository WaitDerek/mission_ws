from glob import glob

from setuptools import find_packages, setup


package_name = "execute_grasp_script_runner"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml", "main.py"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="dekc",
    maintainer_email="dekc@example.com",
    description=(
        "ExecuteGrasp-compatible action server that launches one Python script."
    ),
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "execute_grasp_script_server = "
            "execute_grasp_script_runner.action_server:main",
        ],
    },
)
