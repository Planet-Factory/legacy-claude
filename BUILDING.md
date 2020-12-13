# Running CLAuDE

CLAuDE is built using Python and Cython, a utility which compiles python code into C to speed up execution.

The basic steps to run the program are as follows:

1. Use Cython to compile `claude_low_level_library` and `claude_top_level_library`
```
python claude_setup.py build_ext --inplace
```

2. Run the model
```
python ./toy_model.py
```

Below are more specific instructions to get setup for various operating systems.

## Windows

### 1. Clone or download the repository
If you plan on contributing to this project, you will need to use git in some way to clone the project. Otherwise, simply selecting to download the repo will do.
Various git programs, such as [GitHub Desktop](https://docs.github.com/en/desktop/contributing-and-collaborating-using-github-desktop/cloning-a-repository-from-github-to-github-desktop), [GitKraken](https://support.gitkraken.com/working-with-repositories/open-clone-init/), and [git for Windows](https://gitforwindows.org/) are available to clone the repository with git. Each of these provide different levels of features and different levels of complexity.

### 2. Install Python, pip and a compiler
To build this project, you will need Python, a compiler for the C code that Cython generates, and several python libraries that will be installed with pip.
Pip is a package manager, a program which simplifies installation of software, in this case, python libraries.

You can install python from [its website](https://www.python.org/downloads/windows/).

If you do not already have pip installed (it should be installed with python if you have an up to date version), follow the instructions on the [pip website](https://pip.pypa.io/en/stable/installing/) to install it.

#### Compiler
A compiler must then be installed. According to [the cython documentation](https://cython.readthedocs.io/en/latest/src/quickstart/install.html), you can use either [MinGW](https://osdn.net/projects/mingw/releases/) or Microsoft Visual C.
If you choose to use MinGW, follow the instructions in the [cython documentation](https://cython.readthedocs.io/en/latest/src/tutorial/appendix.html) to get it set up.
If you choose to use Microsoft Visual C/C++, follow the instructions on the [Python website](https://wiki.python.org/moin/WindowsCompilers).

### 3. Use pip to install python libraries
Using the command prompt, install the following pip packages: [cython](https://pypi.org/project/Cython/), [numpy](https://pypi.org/project/numpy/), [matplotlib](https://pypi.org/project/matplotlib/), and [numba](https://pypi.org/project/numba/), as well as [setuptools](https://pypi.org/project/setuptools/), if for some reason it is not already installed.
The syntax to install a package with pip is
```
pip install PACKAGE
```
Where `PACKAGE` is replaced with the name of the package.

### 4. Using Cython, compile `claude_low_level_library` and `claude_top_level_library`
Simply open command prompt, navigate to directory containing this file, and run:
```
python claude_setup.py build_ext --inplace
```
This will convert the `.pyx` files in the repository to C files, then compile them.

### 5. Run the model
Simply run this in your command prompt, while in the directory containing this file.
```
python toy_model.py
```

## Linux

The following programs and libraries are required to build CLAuDE:

 - git (to clone the repository)
 - python
 - setuptools
 - cython
 - numpy
 - matplotlib
 - numba

If you're using apt as your package manager, these correspond to the packages `git` `python3` `python3-setuptools` `cython3` `python3-numpy` `python3-matplotlib` `python3-numba`.

### 1. Clone the repository.
Using the git command line, and cloning via https, this command will do it:
```
git clone https://github.com/Planet-Factory/claude.git
```
If using a GUI front-end for git, your method of doing it may vary.

### 2. Using Cython, compile `claude_low_level_library` and `claude_top_level_library`
In your terminal, in the base `claude` directory, simply run
```
python3 claude_setup.py build_ext --inplace
```
This will convert the `.pyx` files in the repository to C files, then compile them.

### 3. Run the model
In your terminal, run
```
python3 ./toy_model.py
```

#### 2.1 Troubleshooting
For one reason or another, step 2 may fail. Here are some errors you might run in to:

##### Python.h: no such file or directory

This means that you did not install the equivalent of apt's `python3-dev` package - you only have python for running programs, but lack the necessary headers for developing with it. If you are using apt and install the packages listed above, this should not occur.

##### some_other_file.h: no such file or directory

This likely means that the compiler building the libraries is not being told to search the proper directories for header files to include.
One such case where this can happen is when you have libxcrypt installed, so Python.h includes `<crypt.h>`, but setuptools does not tell the compiler where to search for that.
There is probably a proper solution to this, but I don't know it, so for now, you can work around this by telling it manually what to include.
For example, with following error:
```
/usr/include/python3.8/Python.h:44:10: fatal error: crypt.h: No such file or directory
   44 | #include <crypt.h>
      |
```
You need to find where the crypt.h header is to tell setuptools where to include it, so you can use the aptly named `find`.
You can use `find` in the format `find dir/ -name "pattern"` to recursively search directories from `dir/` and get the full path of the header file you are searching for.
If the #include has a directory included (e.g. `#include <blah/foo.h>`), use the directory which contains that directory (if `foo.h` is in `/usr/include/alice/blah/foo.h`, you want `/usr/include/alice`).

So for example, you use `find /usr -name "crypt.h"` to find the full path of the crypt.h header that Python.h is looking to include.
I got the output `/usr/include/tirpc/rpcsvc/crypt.h`, so now I know where that file is found, and I can instruct setuptools to include it.

You instruct setuptools what directories to include with the `-I` option to the command, so in this case, it would be
```
python3 claude_setup.py build_ext --inplace -I/usr/include/tirpc/rpcsvc/
```
Separate multiple includes with `:`, so including `/usr/include/foo` and `/usr/include/bar`, you do `-I/usr/include/foo:/usr/include/bar`.

However, when you manually specify what directories to include, it will not automatically include numpy, so you will need to specify that too. You can wait until it gives you an error about that to find what you need to include.

An actual example command is
```
python claude_setup.py build_ext --inplace -I/usr/include/tirpc:/usr/include/tirpc/rpcsvc:/usr/lib/python3.8/site-packages/numpy/core/include
```

## OSX (not tested)

As stated, these are not tested, but these are the hypothetical steps to set up and build CLAuDE on OSX.
Please report if these do or do not work, so documentation can be updated.

### 1. Install HomeBrew
HomeBrew is a package manager, software which simplifies the installation of various software.
You can install HomeBrew with instructions from [its website](https://brew.sh/).

### 2. Clone the repository
If you plan on contributing to this project, you will need to use git in some way to clone the project. Otherwise, simply selecting to download the repo will do.
Various git programs, such as [GitHub Desktop](https://docs.github.com/en/desktop/contributing-and-collaborating-using-github-desktop/cloning-a-repository-from-github-to-github-desktop), [GitKraken](https://support.gitkraken.com/working-with-repositories/open-clone-init/), or just git are available to clone the repository. Each of these provide different levels of features and different levels of complexity.
git can be installed through homebrew using `brew install git`.
Using git, you can clone the repository via https with this command:
```
git clone https://github.com/Planet-Factory/claude.git
```

### 3. Install dependencies
To build the project, you require `python`, as well as the python libraries `setuptools`, `cython`, `numpy`, `numba` and `matplotlib`.
First, use brew to install python:
```
brew install python
```
Then, use `pip`, a python package manager that is installed with python to install the other dependencies.
`setuptools` should automatically be included with pip, but if it is not, install it along with these packages.
```
pip install cython
pip install numpy
pip install matplotlib
pip install numba
```

### 4. Using Cython, compile `claude_low_level_library` and `claude_top_level_library`
In your terminal, in the base `claude` directory, simply run
```
python claude_setup.py build_ext --inplace
```
This will convert the `.pyx` files in the repository to C files, then compile them.

### 3. Run the model
In your terminal, run
```
python ./toy_model.py
```
