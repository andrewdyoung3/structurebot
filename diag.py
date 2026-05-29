from wsl_bridge import WSLBridge
wsl = WSLBridge()
print('available:', wsl.is_available())
print('check_pyrosetta:', wsl.check_pyrosetta())
r = wsl.run_command('/home/andre/pyrosetta_env/bin/python -c "import pyrosetta; print(chr(79)+chr(75))"')
print('direct run:', r)
