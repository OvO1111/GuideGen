try:
    import tigre
    from tigre.utilities.geometry import Geometry
    from tigre.utilities import gpu, CTnoise
except:
    print('no tigre')
from skimage.transform import resize
import numpy as np, os, sys, SimpleITK as sitk, torch as th, random, shutil

sys.path.append('/mnt/data_1/dlr/code/diffusion/')
sys.path.append('/mnt/data_1/dlr/code/diffusion/dependency/LEAP')

try:
    from leapctype import tomographicModels
    from leaptorch import Projector, FBPReverseFunctionGPU
except:
    print('no leapct')


hypers = {
    'mode': 'set_parallelbeam',
    'filter': None,
    'distance_source_detector': 1400,
    'distance_source_origin': 1100,
    'n_detector': [256, 512],   # (v,u) resolution
    's_detector': [2, 2],     # size of (u,v) 
    'n_voxel': [256, 512, 512],
    's_voxel': [2, 2, 2],
    'offset_origin': [0, 0, 0],
    'offset_detector': [0, 0],
    'accuracy': 0.5,            # of forward projection
    'total_angles': 360,
    'max_angle': 360,
    'start_angle': 0,
    'noise': {'gaussian': {'mean': 0, 'std': 10}, 'poisson': {'mean': 10000}}
}

def forward_projection(file_path: str, save_path: str=None, algo='tigre'):
    # load ct
    if not isinstance(file_path, str): ct = file_path
    else:
        ct = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
        ct = resize(ct, hypers['n_voxel'], order=3, mode='constant', anti_aliasing=True)
    
    # whether cone or parallel projection
    if algo == 'tigre':
        mode = 'parallel' if 'parallel' in hypers['mode'] else 'cone'
        geo = tigre.geometry(mode=mode, nVoxel=np.array(hypers['n_voxel']))
        
        geo.DSD = hypers['distance_source_detector']
        geo.DSO = hypers['distance_source_origin']
        geo.nDetector = np.array(hypers['n_detector'])
        geo.sDetector = np.array(hypers['s_detector'])
        geo.dDetector = geo.sDetector / geo.nDetector
        
        geo.nVoxel = np.array(hypers['n_voxel'])
        geo.sVoxel = np.array(hypers['s_voxel'])
        geo.dVoxel = geo.sVoxel / geo.nVoxel
        geo.offOrigin = np.array(hypers['offset_origin'])
        geo.offDetector = np.array([hypers['offset_detector'][1], hypers['offset_detector'][0], 0])
        geo.accuracy = hypers['accuracy']
        geo.filter = hypers.get('filter')
        
        proj_angles = (
            np.linspace(0, hypers['total_angles'] / 180 * np.pi, 360)[:-1]
            + hypers["start_angle"] / 180 * np.pi
        )
        projs = tigre.Ax(ct + np.abs(ct.min()), geo, proj_angles)[:, ::-1, :]
        
    elif algo == 'leap':
        projector = Projector(forward_project=True, use_static=True, use_gpu=True, gpu_device=th.device('cuda:0'), batch_size=1)
        projector.set_parallelbeam(
            numAngles=hypers['total_angles'],
            numRows=hypers['n_detector'][0], 
            numCols=hypers['n_detector'][1],
            pixelHeight=hypers['s_detector'][0],
            pixelWidth=hypers['s_detector'][1],
            centerRow=hypers['n_detector'][0] // 2,
            centerCol=hypers['n_detector'][1] // 2,
            phis=(
                    np.linspace(0, hypers['max_angle'], hypers['total_angles'])
                + hypers["start_angle"]
            ).astype(np.float32),
            # sod=hypers['distance_source_origin'],
            # sdd=hypers['distance_source_detector'],
        )
        projector.set_volume(
            numX=hypers['n_voxel'][2],
            numY=hypers['n_voxel'][1],
            numZ=hypers['n_voxel'][0],
            voxelWidth=hypers['s_voxel'][0],
            voxelHeight=hypers['s_voxel'][1],
            offsetX=hypers['offset_origin'][0],
            offsetY=hypers['offset_origin'][1],
            offsetZ=hypers['offset_origin'][2],
        )
        
        f = th.tensor(ct + np.abs(ct.min())).contiguous()[None].to(th.device('cuda:0'))
        # f = th.tensor(ct).contiguous()[None].to(th.device('cuda:0'))
        projs = projector(f).cpu().numpy()[0, :, ::-1, ::-1]
    
    if save_path is not None:
        sitk.WriteImage(sitk.GetImageFromArray(projs), save_path)
    
    return np.ascontiguousarray(projs)


def backward_projection(file_path: str, save_path: str=None, algo='tigre'):
    # load projections
    if not isinstance(file_path, str): projs = file_path
    else:
        projs = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
    
    # whether cone or parallel projection
    if algo == 'tigre':
        mode = 'parallel' if 'parallel' in hypers['mode'] else 'cone'
        geo = tigre.geometry(mode=mode, nVoxel=np.array(hypers['n_voxel']))
        
        geo.DSD = hypers['distance_source_detector']
        geo.DSO = hypers['distance_source_origin']
        geo.nDetector = np.array(hypers['n_detector'])
        geo.sDetector = np.array(hypers['s_detector'])
        geo.dDetector = geo.sDetector / geo.nDetector
        
        geo.nVoxel = np.array(hypers['n_voxel'])
        geo.sVoxel = np.array(hypers['s_voxel'])
        geo.dVoxel = geo.sVoxel / geo.nVoxel
        geo.offOrigin = np.array(hypers['offset_origin'])
        geo.offDetector = np.array([hypers['offset_detector'][1], hypers['offset_detector'][0], 0])
        geo.accuracy = hypers['accuracy']
        geo.filter = hypers.get('filter')
        
        proj_angles = (
            np.linspace(0, hypers['total_angles'] / 180 * np.pi, 360)[:-1]
            + hypers["start_angle"] / 180 * np.pi
        )
        ct = tigre.algorithms.fdk(projs, geo, proj_angles)[:, ::-1, :]
        
    elif algo == 'leap':
        projector = Projector(forward_project=False, use_static=True, use_gpu=True, gpu_device=th.device('cuda:0'), batch_size=1)
        projector.set_parallelbeam(
            numAngles=hypers['total_angles'],
            numRows=hypers['n_detector'][0], 
            numCols=hypers['n_detector'][1],
            pixelHeight=hypers['s_detector'][0],
            pixelWidth=hypers['s_detector'][1],
            centerRow=hypers['n_detector'][0] // 2,
            centerCol=hypers['n_detector'][1] // 2,
            phis=(
                    np.linspace(0, hypers['max_angle'], hypers['total_angles'])
                + hypers["start_angle"]
            ).astype(np.float32),
            # sod=hypers['distance_source_origin'],
            # sdd=hypers['distance_source_detector'],
        )
        projector.set_volume(
            numX=hypers['n_voxel'][2],
            numY=hypers['n_voxel'][1],
            numZ=hypers['n_voxel'][0],
            voxelWidth=hypers['s_voxel'][0],
            voxelHeight=hypers['s_voxel'][1],
            offsetX=hypers['offset_origin'][0],
            offsetY=hypers['offset_origin'][1],
            offsetZ=hypers['offset_origin'][2],
        )
        # projector.leapct.set_diameterFOV(1e16)
        projector.leapct.set_truncatedScan(True)
        
        f = th.tensor(projs).contiguous()[None].to(th.device('cuda:0'))
        # dropout = th.randperm(f.shape[1])[:300]
        # f[:, dropout] = 0
        ct = projector.fbp(f).cpu().numpy()[0, ::-1, ...] + (-1024)  # add -1024 to represent actual air HU value
        FBPReverseFunctionGPU.apply(f, projector.proj_data, projector.vol_data, projector.param_id_t)
    
    if save_path is not None:
        sitk.WriteImage(sitk.GetImageFromArray(ct), save_path)
    
    return np.ascontiguousarray(ct)


def projection(file_path: str, algo='tigre', save_path_proj=None, save_path_ct=None):
    shutil.copyfile(file_path, os.path.join(os.path.dirname(save_path_proj), 'raw.nii.gz'))
    proj = forward_projection(file_path, algo=algo, save_path=save_path_proj)
    ct = backward_projection(proj, save_path_ct, algo=algo)
    return proj, ct


def pipeline():
    base = '/mnt/data_1/dlr/data/datasets'
    for dataset in os.listdir(base):
        dirname = os.path.join(base, dataset, 'imagesTs')
        os.makedirs(os.path.join(base, dataset, 'projsTs'), exist_ok=True)
        for file in os.listdir(dirname):
            forward_projection(os.path.join(dirname, file), os.path.join(base, dataset, 'projsTs', file))


if __name__ == '__main__':
    # forward_projection('/mnt/data_1/dlr/data/datasets/msd_liver/imagesTs/liver_133.nii.gz', algo='leap', save_path='./projs.nii.gz')
    # backward_projection('./projs.nii.gz', algo='leap', save_path='./ct.nii.gz')
    projection('/mnt/data_1/dlr/data/datasets/msd_liver/imagesTs/liver_133.nii.gz', algo='tigre', save_path_proj='/home/dlr/data/cache/projs.nii.gz', save_path_ct='/home/dlr/data/cache/ct.nii.gz')