import gurobipy as gp
from gurobipy import GRB

import gpuMonitor
import numpy as np

if __name__ == "__main__":

    try:
        # Create a new model
        m = gp.Model("mip1")

        # Create variables
        x = m.addVar(vtype=GRB.BINARY, name="x")
        y = m.addVar(vtype=GRB.BINARY, name="y")
        z = m.addVar(vtype=GRB.BINARY, name="z")

        # Add constraint: x + 2 y + 3 z <= 4
        m.addConstr(x + 2 * y + 3 * z <= 4, "c0")

        # Add constraint: x + y >= 1
        m.addConstr(x + y >= 1, "c1")

        # Set objective
        m.setObjective(x + y + 2 * z, GRB.MAXIMIZE)

        if gpuMonitor.is_gpu_available():
            gpuMonitor.start_monitoring(interval=0.01)

            # Optimize model
            m.optimize()

        if gpuMonitor.is_gpu_available():
            gpu_usage = gpuMonitor.gpu_stats  # get the collected gpu usage stats
            if gpu_usage:
                print(f"GPU usage (extract): {gpu_usage[0:10]}")
                # Calculate average and max GPU usage
                average_gpu_usage = np.mean(gpu_usage)
                max_gpu_usage = np.max(gpu_usage)
                print(f"Average GPU usage: {average_gpu_usage}")
                print(f"Maximum GPU usage: {max_gpu_usage}")
            else:
                # Handle the case where gpu_usage is empty
                print("Warning: GPU usage data is empty.")
        else:
            # No GPU
            print("No GPU available")

        for v in m.getVars():
            print('%s %g' % (v.VarName, v.X))

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))
    except AttributeError:
        print('Encountered an attribute error')