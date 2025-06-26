"""Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
from setuptools import find_packages, setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.develop import develop as _develop


class ResiliencyDevelop(_develop):

  def run(self):
    link_path = os.path.join("resiliency", "third_party")
    target_path = os.path.join("..", "third_party")
    if not os.path.exists(link_path):
      print(f"Creating symlink: {link_path} → {target_path}")
      os.symlink(target_path, link_path)
    super().run()


class ResiliencyBuildPy(_build_py):

  def run(self):
    # Create resiliency/third_party → ../third_party if it doesn't exist
    link_path = os.path.join("resiliency", "third_party")
    target_path = os.path.join("..", "third_party")
    if not os.path.exists(link_path):
      print(f"Creating symlink: {link_path} → {target_path}")
      os.symlink(target_path, link_path)
    super().run()


setup(
    name="resiliency",
    version="0.1.0",
    packages=find_packages(include=["resiliency*"]),
    include_package_data=True,
    zip_safe=False,
    cmdclass={
        "build_py": ResiliencyBuildPy,
        "develop": ResiliencyDevelop,
    },
)
