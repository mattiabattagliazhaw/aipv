from numpy import random
from lib.laoss_interface import *
from functools import partial
from shutil import copyfile
import pathlib
import tqdm

file = pathlib.Path(__file__).absolute()

random.seed(11)

def create_image(i,path=None):
    config_file = "./lib/source_files/config.json"
    with open(r"./lib/source_files/base_values_laoss.json") as file:
        instance = json.load(file)
    instance['index'] = i
    random.seed(i)
    if i != 0:

        instance['V1']= random.uniform(0.54,0.55)
        instance['V2'] = random.uniform(0.60, 0.65)
        instance['dark_saturation_current_0'] = loguniform.rvs(1e-10, 1e-8, size=1)[0]
        instance["n_shunts"] = random.randint(0, 5)
        instance['r_sheet_0'] = np.random.uniform(10, 120)
        instance['r_sheet_1'] = loguniform.rvs(1e-4, 1e-2, size=1)[0]
        instance['shunt_positions'] = []
        for z in range(instance["n_shunts"]):
            for j in range(1000):
                instance['shunt_positions'].append([np.random.uniform(0.6, 1.3), np.random.uniform(0.13, 1.72)]) #0.13
            instance['dark_saturation_current_' + str(z + 2)] = instance['dark_saturation_current_0']
            instance['r_sheet_' + str(z + 2)] = 10
            instance['r_int_' + str(z + 2)] = 1e-5
            instance['rho_par_'+ str(z + 2)] = loguniform.rvs(1e3, 2e6, size=1)[0]



    laoss_run = LaossRun(path,None,None)
    laoss_config=laoss_run.prepare(config_file,instance)
    sweeps = []
    sweeps.append(
        {'conditions': ['edge31'], 'id': "Condition.Dirichlet.Potential", 'target': "top", 'type': "linear",
         'values': {'range': {"end": instance['V2'], "start": instance['V1'], "step": (instance['V2']-instance['V1'])}, 'unit': "V"}})
    sweeps.append(
        {"conditions": ["electrode", "Metal"], "id": "CouplingElectric.LambdaJVCurve.Wavelength", "target": "both",
         "type": "linear", "values": {"range": {"end": 0.6, "start": 0.0, "step": 1}, "unit": "m"}})
    laoss_config.set_sweep(sweeps)
    laoss_config.save_config()
    return instance

if __name__ == '__main__':
    script_path = str(pathlib.Path(__file__).parent.absolute())
    path = os.path.join(script_path,r'../example_data')
    n_parallel = 4
    laoss_kernel_path = "C:\Program Files\Fluxim\LAOSS 4.1\laoss-kernel.exe"
    laoss_tool_path = "C:\Program Files\Fluxim\LAOSS 4.1\laoss-tool.exe"
    n_images = 100
    script_path = str(pathlib.Path(__file__).parent.absolute())
    run = LaossRun(path,laoss_kernel_path,laoss_tool_path,overwrite=True,noise_folder=os.path.join(script_path,r"noise_samples"))
    run_list = []
    pool = Pool(processes=n_parallel)
    for result in tqdm.tqdm(pool.imap_unordered(partial(create_image,path=path), range(n_images)), total=n_images):
       run_list.append(result)
    pool.close()

    run_paths = run._run_and_process_parallel_progress_bar(run_list,n_processes=n_parallel,clean=True)

    copyfile(file,os.path.join(path,'run_script.py'))
    pd.DataFrame(run_list).to_csv(os.path.join(path,'run_list.csv'))

