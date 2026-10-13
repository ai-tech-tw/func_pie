from setuptools import setup, find_packages

setup(
    name="funcpie",
    version="1.0.0",
    packages=find_packages(include=["funcpie*", "func_pie*", "libs*"]),
    package_data={"funcpie": ["py.typed"]},
    include_package_data=True,
    python_requires=">=3.8",
)
