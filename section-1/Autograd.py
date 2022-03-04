import torch

if __name__ == "__main__":
    # torch.autograd를 사용한 자동 미분
    x = torch.ones(5) # input tensor
    y = torch.zeros(3) # expected output
    w = torch.randn(5, 3, requires_grad=True)
    b = torch.randn(3, requires_grad=True)
    z = torch.matmul(x, w) + b
    loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

    # 변화도 계산하기
    loss.backward()
    print(w.grad)
    print(b.grad)

    # 변화도 추적 멈추기
    # 1
    z = torch.matmul(x, w) + b
    print(z.requires_grad)

    with torch.no_grad():
        z = torch.matmul(x, w) + b
    print(z.requires_grad)
    # 2
    z = torch.matmul(x, w) + b
    z_det = z.detach()
    print(z_det.requires_grad)

    # 선택적으로 읽기: 텐서 변화도와 야코비안 곱
    inp = torch.eye(5, requires_grad=True)
    out = (inp + 1).pow(2)
    out.backward(torch.ones_like(inp), retain_graph=True)
    print("First call\n", inp.grad)
    out.backward(torch.ones_like(inp), retain_graph=True)
    print("\nSecond call\n", inp.grad)
    inp.grad.zero_()
    out.backward(torch.ones_like(inp), retain_grah=True)
    print("\nCall after zeroing gradients\n", inp.grad)