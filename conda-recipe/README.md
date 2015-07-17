To build the ttpy module

1. Install the pre requirements
    ```
   conda install conda-build
   ```
2. Add tag to the release on github;
3. Obtain a tar archive with all the ttpy code (including the submodules);
4. Load the archive to the web;
5. Edit the release version, .tar archive url and its md5 check sum in the ```meta.yaml```;
6. Build the module:
   ```
   conda build conda-recipe
   ```
7. Try to install it locally:
   ```
   conda install --use-local conda-recipe
   ```
8. Upload the module to the binstar:
```
binstar login
binstar upload /home/alex/anaconda/conda-bld/linux-64/ttpy-1.0-np19py27_0.tar.bz2  
binstar logout
```
