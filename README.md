# MagPy

This project is a magnifier application that displays a magnified view of the area around the mouse cursor. The ideal use case is for streamers or content creators that do a lot of instructional work on computers. This avoids having to use display scaling to make the application more visible via stream or video and provides an easy means for both screen recording and live streams to provide detail when teaching complex apps or concepts.

## Features

- User configurable constants in script; defaults below.
- 3x magnification 
- 16:9 aspect ratio window (800x450)
- Random repositioning to avoid mouse overlap
- Always on top
- Automatically hides the magnifier window after 3 seconds of inactivity
- Displays keystrokes on the top left of the screen, including modifier keys
- Each keystroke entry fades out after 1 second

## Prerequisites
 - A CUDA 11 or greater enabled GPU
 - CUDA toolkit installed.

## Installation

1. Download the archive from the [releases](https://github.com/jhancuff/mag-gpu/releases) page.
2. Unzip anywhere.
3. Run the executable on your Windows machine.

## Usage

1. Run the `mag-gpu.exe` file.
2. Move your mouse around to see the magnified view.
3. Push the mouse into the window to have it dodge your cursor.
4. The magnifier window will automatically hide after 3 seconds of inactivity and reappear when the mouse is moved.

## Demonstration
[![Watch the video](https://img.youtube.com/vi/33MllWdQwxo/maxresdefault.jpg)](https://www.youtube.com/watch?v=33MllWdQwxo)

## Building from Source

1. Ensure you have Python and the required libraries installed.
2. Clone the repository and navigate to the project directory.
3. Install dependencies using `pip install -r requirements.txt`.
4. Run the Python script using `python src/mag-gpu.py`.

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE. See the [LICENSE](LICENSE) file for details.
