from marketmaker import GaussianHMM, tensor_convert, retrieve_data, normalize_tensor
import torch


def main():
    tensor = tensor_convert(retrieve_data("TSLA"))
    norm_tensor = normalize_tensor(tensor)
    ghmm = GaussianHMM(
        5,
        5,
    )
    T = torch.full((10,), 365)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    ghmm = ghmm.to(device)

    response = ghmm.baum_welch(norm_tensor, T)

    print(response)


if __name__ == "__main__":
    main()
