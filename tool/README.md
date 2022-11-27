
1. Clone and install EasyMocap:
   ```
https://chingswy.github.io/easymocap-public-doc/install/install.html
   ```

2. copy `get_X_pose.py` to EasyMocap/apps/demo
  
3. modify the `data_base` and run the script

It is used to compute the X-posed SMPL model in the canonical space using individual `shapes` params.

I provide a sample X_smpl_vertices.npy in the dictionary for performer 313 in Zju-Mocap, but please note that each performer don not share the same `shapes` params.