r"""gpu_test.py
Petit utilitaire de test pour vérifier si OpenCV a un backend CUDA utilisable.
Usage: python src\gpu_test.py
"""

import sys
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None


def is_opencv_cuda_available():
    if cv2 is None:
        return False
    try:
        if hasattr(cv2, 'cuda'):
            try:
                return int(cv2.cuda.getCudaEnabledDeviceCount() or 0) > 0
            except Exception:
                pass
        info = cv2.getBuildInformation()
        return 'CUDA' in info
    except Exception:
        return False


def main():
    print('Python:', sys.version.splitlines()[0])
    if cv2 is None:
        print('OpenCV (cv2) non installé dans cet environnement.')
        return
    print('OpenCV version:', cv2.__version__)
    has_cuda = is_opencv_cuda_available()
    print('OpenCV CUDA disponible (build+device):', has_cuda)

    if not has_cuda:
        print('\nRemarques:')
        print('- Si vous voulez utiliser GPU, installez une build d\'OpenCV compilée avec CUDA ou installez opencv-python-headless-cuda (si disponible).')
        print('- Alternativement, utilisez ffmpeg pour décoder puis envoyez les frames sur GPU via dnn/cuvid si nécessaire (workflow avancé).')
        return

    # Test simple: upload, convert to gray on GPU, download, compare with CPU result
    try:
        count = int(cv2.cuda.getCudaEnabledDeviceCount() or 0)
    except Exception:
        count = 0
    print('Nombre de devices CUDA détectés:', count)
    try:
        cv2.cuda.setDevice(0)
    except Exception:
        pass

    h, w = 240, 320
    cpu_img = (np.random.randint(0, 256, (h, w, 3), dtype=np.uint8))
    try:
        gpu_mat = cv2.cuda_GpuMat()
        gpu_mat.upload(cpu_img)
        # try cvtColor on GPU
        try:
            gpu_gray = cv2.cuda.cvtColor(gpu_mat, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            print('Erreur lors de cv2.cuda.cvtColor:', e)
            return
        result = gpu_gray.download()
        cpu_gray = cv2.cvtColor(cpu_img, cv2.COLOR_BGR2GRAY)
        # compare
        diff = int(np.max(np.abs(result.astype(np.int32) - cpu_gray.astype(np.int32))))
        print('Max difference between GPU and CPU gray conversion:', diff)
        if diff == 0:
            print('Succès: opération GPU cohérente avec CPU.')
        else:
            print('Attention: différences observées, mais GPU fonctionne probablement.')
    except Exception as e:
        print('Erreur générale en utilisant cv2.cuda:', e)


if __name__ == '__main__':
    main()
