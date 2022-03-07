import torch
import torchvision.models as models

# 모델 저장하고 불러오기
if __name__ == "__main__":
    # 모델 가중치 저장하고 불러오기
    model = models.vgg16(pretrained=True)
    torch.save(model.state_dict(), 'model_weights.pth')

    model = models.vgg16() # 기본 가중치를 불러오지 않으므로 pretrained=True를 지정하지 않는다.
    model.load_state_dict(torch.load('model_weights.pth'))
    model.eval() # 드롭아웃과 배치 정규화를 평가 모드로 설정

    # 모델의 형태를 포함하여 저장하고 불러오기
    torch.save(model, 'model.pth')
    model = torch.load('model.pth')