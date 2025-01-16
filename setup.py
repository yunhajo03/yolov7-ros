from setuptools import find_packages, setup

package_name = 'yolov7_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/yolov7.launch'])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yunhajo',
    maintainer_email='yjo@caltech.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detect_ros = yolov7_ros.detect_ros:main'
        ],
    },
)
