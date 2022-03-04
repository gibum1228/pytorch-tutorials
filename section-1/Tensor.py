import torch
import numpy as np

if __name__ == "__main__":
    """
    tensor 초기화
    """
    #1 데이터로부터 직접 생성하기
    data = [[1, 2], [3, 4]]
    x_data = torch.tensor(data)
    # numpy 배열로부터 생성하기
    np_array = np.array(data)
    x_np = torch.from_numpy(np_array)
    # 다른 텐서로부터 생성하기
    x_ones = torch.ones_like(x_data)  # x_data의 속성을 유지합니다.
    print(f"Ones Tensor: \n {x_ones} \n")

    x_rand = torch.rand_like(x_data, dtype=torch.float)  # x_data의 속성을 덮어씁니다.
    print(f"Random Tensor: \n {x_rand} \n")
    # 무작위 또는 상수 값을 사용하기
    shape = (2, 3,)
    rand_tensor = torch.rand(shape)
    ones_tensor = torch.ones(shape)
    zeros_tensor = torch.zeros(shape)

    print(f"Random Tensor: \n {rand_tensor} \n")
    print(f"Ones Tensor: \n {ones_tensor} \n")
    print(f"Zeros Tensor: \n {zeros_tensor}")

    """
    텐서의 속성
    """
    tensor = torch.rand(3, 4)

    print(f"Shape of tensor: {tensor.shape}")
    print(f"Datatype of tensor: {tensor.dtype}")
    print(f"Device tensor is stored on: {tensor.device}")

    """
    텐서 연산
    """
    # GPU가 존재하면 텐서를 이동합니다
    if torch.cuda.is_available():
        tensor = tensor.to('cuda')

    # Numpy식의 표준 인덱싱과 슬라이싱
    tensor = torch.ones(4, 4)
    print('First row: ', tensor[0])
    print('First column: ', tensor[:, 0])
    print('Last column:', tensor[..., -1])
    tensor[:, 1] = 0
    print(tensor)

    # 텐서 합치기
    t1 = torch.cat([tensor, tensor, tensor], dim=1)
    print(t1)

    # 산술 연산
    # 두 텐서 간의 행렬 곱(matrix multiplication)을 계산합니다. y1, y2, y3은 모두 같은 값을 갖습니다.
    y1 = tensor @ tensor.T
    y2 = tensor.matmul(tensor.T)

    y3 = torch.rand_like(tensor)
    torch.matmul(tensor, tensor.T, out=y3)

    # 요소별 곱(element-wise product)을 계산합니다. z1, z2, z3는 모두 같은 값을 갖습니다.
    z1 = tensor * tensor
    z2 = tensor.mul(tensor)

    z3 = torch.rand_like(tensor)
    torch.mul(tensor, tensor, out=z3)

    # 텐서의 모든 값을 하나로 집계
    agg = tensor.sum()
    agg_item = agg.item()
    print(agg_item, type(agg_item))

    # 바꿔치기 연산
    print(tensor, "\n")
    tensor.add_(5)
    print(tensor)

    """
    NumPy 변환
    """
    # 텐서를 넘파이 배열로 변환
    t = torch.ones(5)
    print(f"t: {t}")
    n = t.numpy()
    print(f"n: {n}")
    # 텐서의 변경 사항이 넘파이 배열에 반영
    t.add_(1)
    print(f"t: {t}")
    print(f"n: {n}")

    # 넘파이 배열을 텐서로 변환
    n = np.ones(5)
    t = torch.from_numpy(n)
    # 넘파이 배열의 변경 사항이 텐서에 반영
    np.add(n, 1, out=n)
    print(f"t: {t}")
    print(f"n: {n}")