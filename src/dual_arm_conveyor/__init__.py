"""Vision-guided, variable-speed conveyor sorting with two or four Panda arms."""

import xml.etree.ElementTree as _ET

# Initialize Python's native XML parser before MuJoCo can load EGL/GLFW.
# On Conda, graphics libraries can otherwise load an older system libexpat
# with the same SONAME first, leaving pyexpat with unresolved symbols.
# This keeps library selection local to this process; no loader paths change.
_ET.XMLParser()

__version__ = "0.8.6"
