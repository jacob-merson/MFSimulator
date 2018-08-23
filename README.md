***********************************************************************************
**  MFSimulator package for simulating fluid-body interactions in potential flow **
***********************************************************************************

(C) Jacob Merson 2018

***********************************************************************************
1. Requirements
***********************************************************************************

To install:
  - python v3.x
- To save animations to file:
  - imagemagick image software suite      (https://www.imagemagick.org/script/download.php)
  - ffmpeg multimedia software suite      (https://www.ffmpeg.org/download.html)
  
  
***********************************************************************************
2. Installation
***********************************************************************************

** The MFSimulator module has been designed for and extensively tested on macOS **
** out of the box support for other OSes- particularly Windows- cannot be guaranteed! **

To install the MFSimulator module,

  - either git clone the repository or download directly to local machine
  - navigate to downloaded root 'MFSimulator' folder at the command line
  - ensure python v3.x is installed (can download at https://www.python.org/downloads/)
  - at the command line, ensuring you are in the root MFSimulator folder (the one with the setup.py file), type:
        
        python3 setup.py install
        
  - or possibly (depending on the alias configuration of installed python versions)
        
        python setup.py install
        
  The MFSimulator package, along with all its dependencies, will be automatically installed
  and the MFSimulator package will be available for system-wide use
  
To install imagemagick and ffmpeg: (needed for saving animations to file)

  MacOS:
    it is recommended to use the Homebrew package manager for MacOS to install
    imagemagick and ffmpeg
    
    - Homebrew can be downloaded at: https://brew.sh/
    
    - Once Homebrew is installed, run at the command line:
        
        brew install imagemagick
        brew install ffmpeg
        
        
  Linux:
    
    At the command line:
    
      sudo apt-get update
      sudo apt-get install imagemagick    (more help available at https://help.ubuntu.com/community/ImageMagick)

    - If using Ubuntu 18.04 or 16.04:
        sudo add-apt-repository ppa:jonathonf/ffmpeg-3
      
    - If using Ubuntu 14.04:
        sudo add-apt-repository ppa:jonathonf/tesseract
        
      sudo apt-get update
      sudo apt-get install ffmpeg libav-tools x264 x265   (more help available at https://tecadmin.net/install-ffmpeg-on-linux/)
      
      
  Windows:
  
    Not recommended, attempt to install ImageMagick and ffmpeg correctly at own risk:
    
    ** installing just ImageMagick SHOULD also install ffmpeg **
    
    Download the ImageMagick Windows Binary (http://www.imagemagick.org/script/download.php#windows)
    
    unpack and install to Program Files
    
    Need to add the paths to ffmpeg.exe and convert.exe in the environment variables list
    to be able to use them at the command line, BUT, do not need to for use with the
    MFSimulator package. Will, however, need to make sure the paths to ffmpeg.exe and convert.exe
    are set correctly in the 'simulator.py' module
    
    Additional information about installing ffmpeg is available at http://blog.gregzaal.com/how-to-install-ffmpeg-on-windows/
    ffmpeg should come as part of ImageMagick on Windows, and the above link is outdated by a few years, but the instructions on how to set environmental variables paths are still valid (and may require a PC restart before they take effect)
    
    
***********************************************************************************
3. Usage
***********************************************************************************
      
To check if the package has installed correctly, load a python session from the command line
by typing 'python3' at the command line and hitting enter

Once in a python 3 shell, attempt to import the following module:

  from MFSimulator.simulator import Model
  
If the Model class imports, then the package has been installed correctly (type quit() to exit python shell)

The entire program is operated by the Model class in the simulator module

To import in your own program scripts, add the line
  
  from MFSimulator.simulator import Model
  
And then instantiate Model objects with Model(...)

where the ... are the relevant system model parameters (see class documentation)

Demo files are included. To operate,

  - move to the 'demos' directory in the MFSimulator folder (in the root MFSimulator folder)
  - will see THREE python script files and FOUR yml text files
  
  The yml text files consist of dictionaries that define the simulation parameters
  (i.e. they are example system configurations)
  
  The .py script files are used to demonstrate the program's capabilities
  
  To run the demos, making sure you are in the 'demos' directory, run (for example):
  
    python3 StaticFlowVisualization.py --model one-circle-circulation.yml
  
  which, in this case, will demonstrate the program's static flow field visualization
  capabilities for a fluid-body system consisting of one circle in unbounded flow with
  nonzero circulation
  
  To run the additional feature demo files (DynamicMotionAnimation.py, SaveAnimationDemo.py)
  and/or other configuration example files (two-circle-collision.yml, two-circle-collision.yml, one-circle-wall.yml),
  
  use the same syntax as above, replacing 'StaticFlowVisualization.py' with the desired demo name
  and 'one-circle-circulation.yml' with the desired system configuration
  
  IF you encounter problems saving the animations to file, ensure that the (OS appropriate) paths
  to convert and ffmpeg at the top of the simulator module (lines 34-46) are correct for your
  ffmpeg and convert install locations
  
  It is highly recommended that animations be saved to mp4 for the smoothest, true to frame-rate
  playback speed
  
Enjoy!
  
  
***********************************************************************************
4. Uninstall
***********************************************************************************
  
  TO UNINSTALL:
    - at the command line, type:
        pip3 uninstall MFSimulator
      
    - or, possibly (depending on python version/path aliases)
      
        pip uninstall MFSimulator
      
      
























