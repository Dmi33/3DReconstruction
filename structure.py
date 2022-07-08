import numpy as np

def reconstruct_points(p1, p2, m1, m2):
    num_points = p1.shape[1]
    res = np.ones((4, num_points))

    for i in range(num_points):
        res[:, i] = reconstruct_one_point(p1[:, i], p2[:, i], m1, m2)

    return res


def reconstruct_one_point(pt1, pt2, m1, m2):
    """
    pt1 x m1 * X  =  pt2 x m2 * X  =  0
    """
    A = np.vstack([
        np.dot(skew(pt1), m1),
        np.dot(skew(pt2), m2)
    ])
    U, S, V = np.linalg.svd(A)
    P = np.ravel(V[-1, :4])

    return P / P[3]


def linear_triangulation(p1, p2, m1, m2):
    """
    p1, p2: 2D координаты (3 x n)
    m1, m2: матрицы камеры для p1 и p2 (3 x 4)
    Возвращает  4 x n 3d координаты
    """
    num_points = p1.shape[1]
    res = np.ones((4, num_points))

    for i in range(num_points):
        A = np.asarray([
            (p1[0, i] * m1[2, :] - m1[0, :]),
            (p1[1, i] * m1[2, :] - m1[1, :]),
            (p2[0, i] * m2[2, :] - m2[0, :]),
            (p2[1, i] * m2[2, :] - m2[1, :])
        ])

        _, _, V = np.linalg.svd(A)
        X = V[-1, :4]
        res[:, i] = X / X[3]

    return res


def compute_epipole(F):
    """ Вычисляет правый эпиполь из матрицы F
        ( F.T для левого эпиполя)
    """
   
    U, S, V = np.linalg.svd(F)
    e = V[-1]
    return e / e[2]

def skew(x):
    """ 
    Создаёт кососимметричную матрицу A из 3d-вектора x
     np.cross(A, v) == np.dot(x, v)
   
    """
    return np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ])


def compute_P_from_essential(E):
    """ 
    Вычисляет вторую матрицу камеры (предполагая, что P1 = [I 0])
    из матрицы E = [t]R
    
    """
    U, S, V = np.linalg.svd(E)

    # Нужно убедиться, что матрица вращения является правой с положительным определителем
    if np.linalg.det(np.dot(U, V)) < 0:
        V = -V

    
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    P2s = [np.vstack((np.dot(U, np.dot(W, V)).T, U[:, 2])).T,
          np.vstack((np.dot(U, np.dot(W, V)).T, -U[:, 2])).T,
          np.vstack((np.dot(U, np.dot(W.T, V)).T, U[:, 2])).T,
          np.vstack((np.dot(U, np.dot(W.T, V)).T, -U[:, 2])).T]

    return P2s


def correspondence_matrix(p1, p2):
    p1x, p1y = p1[:2]
    p2x, p2y = p2[:2]

    return np.array([
        p1x * p2x, p1x * p2y, p1x,
        p1y * p2x, p1y * p2y, p1y,
        p2x, p2y, np.ones(len(p1x))
    ]).T

    return np.array([
        p2x * p1x, p2x * p1y, p2x,
        p2y * p1x, p2y * p1y, p2y,
        p1x, p1y, np.ones(len(p1x))
    ]).T


def compute_image_to_image_matrix(x1, x2, compute_essential=False):
    """ 
    Вычисляет фундаментальную матрицу из соответствующих точек
    (x1, x2 3 * n ) с использованием 8-точечного алгоритма.
    Каждая строка в приведенной ниже матрице A строится следующим образом
    [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1]
    """
    A = correspondence_matrix(x1, x2)
    # вычислить линейное решение методом наименьших квадратов
    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    # Ограничение F. Ранг должен быть равен 2
    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    if compute_essential:
        S = [1, 1, 0] 
    F = np.dot(U, np.dot(np.diag(S), V))

    return F


def scale_and_translate_points(points):
    """ 
    Масштабируем и переносим точки изображения так, чтобы центроид точек
    находился в начале координат, а среднее расстояние до начала координат равно sqrt(2).
    """
    x = points[0]
    y = points[1]
    center = points.mean(axis=1)  
    cx = x - center[0] 
    cy = y - center[1]
    dist = np.sqrt(np.power(cx, 2) + np.power(cy, 2))
    scale = np.sqrt(2) / dist.mean()
    norm3d = np.array([
        [scale, 0, -scale * center[0]],
        [0, scale, -scale * center[1]],
        [0, 0, 1]
    ])

    return np.dot(norm3d, points), norm3d


def compute_normalized_image_to_image_matrix(p1, p2, compute_essential=False):
    """ 
    Вычисляет фундаментальную матрицу из соответствующих точек,
    используя нормализованный 8-точечный алгоритм 
   
    """
    n = p1.shape[1]
    if p2.shape[1] != n:
        raise ValueError('Number of points do not match.')

    
    p1n, T1 = scale_and_translate_points(p1)
    p2n, T2 = scale_and_translate_points(p2)

    F = compute_image_to_image_matrix(p1n, p2n, compute_essential)
    F = np.dot(T1.T, np.dot(F, T2))

    return F / F[2, 2]


def compute_fundamental_normalized(p1, p2):
    return compute_normalized_image_to_image_matrix(p1, p2)


def compute_essential_normalized(p1, p2):
    return compute_normalized_image_to_image_matrix(p1, p2, compute_essential=True)
