import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import transformer_model as tm


def prep_data():
    dirname = os.path.dirname(__file__)
    # filename = os.path.join(dirname, '..', '..', 'data', 'input_behavior', 'BehaviorScenario_ActivityProfile.xlsx')
    filename = os.path.join(dirname, '..', '..', 'data', 'input_behavior', 'duration-probability.xlsx')
    df = pd.read_excel(filename)
    df = df.drop(columns=['id_hhx', 'id_persx', 'id_tagx', 'monat', 'wtagfei'])

    data = [row for row in df.to_numpy()]

    # Split Data in Training, Validation and Test Data
    n_data = len(data)
    splits = [int(n_data*0.7), int((n_data-int(n_data*0.7))*0.15)]
    training_data = data[:splits[0]]
    validation_data = data[splits[0]:splits[0]+splits[1]]
    test_data = data[splits[0]+splits[1]:]

    #
    training_data = [np.split(x, [int(len(x) / 2)]) for x in training_data]
    training_data = [[np.insert(x, 0, 0) for x in pair] for pair in training_data]
    training_data = [[np.append(x, 18) for x in pair] for pair in training_data]

    validation_data = [np.split(x, [int(len(x) / 2)]) for x in validation_data]
    validation_data = [[np.insert(x, 0, 0) for x in pair] for pair in validation_data]
    validation_data = [[np.append(x, 18) for x in pair] for pair in validation_data]

    test_data = [np.split(x, [int(len(x) / 2)]) for x in test_data]
    test_data = [[np.insert(x, 0, 0) for x in pair] for pair in test_data]
    test_data = [[np.append(x, 18) for x in pair] for pair in test_data]

    return training_data, validation_data, test_data


def batchify_data(data, batch_size=16, padding=False, padding_token=-1):  # batch size in paper 128
    batches = []
    for idx in range(0, len(data), batch_size):
        # We make sure we dont get the last bit if its not batch_size size
        if idx + batch_size < len(data):
            # Here you would need to get the max length of the batch,
            # and normalize the length with the PAD token.
            if padding:
                max_batch_length = 0

                # Get longest sentence in batch
                for seq in data[idx: idx + batch_size]:
                    if len(seq) > max_batch_length:
                        max_batch_length = len(seq)

                # Append X padding tokens until it reaches the max length
                for seq_idx in range(batch_size):
                    remaining_length = max_batch_length - len(data[idx + seq_idx])
                    data[idx + seq_idx] += [padding_token] * remaining_length

            batches.append(np.array(data[idx: idx + batch_size]).astype(np.int64))

    print(f"{len(batches)} batches of size {batch_size}")

    return batches


def train_model(model_name):
    train_data, val_data, test_data = prep_data()

    train_dataloader = batchify_data(train_data)
    val_dataloader = batchify_data(val_data)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = tm.Transformer(
        num_tokens=19, dim_model=8, num_heads=2, num_encoder_layers=3, num_decoder_layers=3, dropout_p=0.1
    ).to(device)  # in paper dim_model=144, num_heads=3, num_encoder_layers=300, num_decoder_layers=0
    opt = torch.optim.Adamax(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    train_loss_list, validation_loss_list = tm.fit(model, opt, loss_fn, train_dataloader, val_dataloader, 10, device)

    dirname = os.path.dirname(__file__)
    P = os.path.join(dirname, '..', '..', 'data', 'output', f'{model_name}.pth.tar')
    torch.save(model.state_dict(), P)

    plt.plot(train_loss_list, label="Train loss")
    plt.plot(validation_loss_list, label="Validation loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.legend()
    plt.show()


def predict(model_name):
    train_data, val_data, test_data = prep_data()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = tm.Transformer(
        num_tokens=19, dim_model=8, num_heads=2, num_encoder_layers=3, num_decoder_layers=3, dropout_p=0.1
    ).to(device)  # in paper dim_model=144, num_heads=3, num_encoder_layers=300, num_decoder_layers=0
    opt = torch.optim.Adamax(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    dirname = os.path.dirname(__file__)
    P = os.path.join(dirname, '..', '..', 'data', 'output', f'{model_name}.pth.tar')
    model.load_state_dict(torch.load(P))

    examples = []
    for data in test_data:
        examples.append(torch.tensor([data[0]], dtype=torch.long, device=device))

    for idx, example in enumerate(examples):
        result = tm.predict(model, example, device)
        print(f"Example {idx}")
        print(f"Input: {example.view(-1).tolist()[1:-1]}")
        print(f"Continuation: {result[1:-1]}")
        print()
