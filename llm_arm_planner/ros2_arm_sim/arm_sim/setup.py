from setuptools import find_packages, setup

package_name = 'arm_sim'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='adithi',
    maintainer_email='you@example.com',
    description='Simulated 3-DOF arm exposing MoveTo/Gripper ROS2 actions',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'arm_sim_node = arm_sim.arm_sim_node:main',
        ],
    },
)
