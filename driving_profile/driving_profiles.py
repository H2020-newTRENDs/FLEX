import pandas as pd


def func(f_name):
    df = pd.read_csv(f_name + ".csv", header=None)
    ndarray = df.to_numpy()
    profile_dict = {}
    for profile in ndarray:
        profile_dict[str(int(profile[0]))] = list(profile[1:]) * 52 + list(profile[1:25])
    df = pd.DataFrame.from_dict(profile_dict)
    print(f"Saving {f_name} to csv...")
    df.to_csv(f_name + "_update.csv", index=False)


def create_outside_charge():
    commercial_charge = pd.read_csv("EVC_1_2030_update.csv")
    work_charge = pd.read_csv("EVC_2_2030_update.csv")
    public_charge = pd.read_csv("EVC_3_2030_update.csv")
    outside_charge = {}
    for profile_id in commercial_charge.columns:
        outside_charge[str(profile_id)] = commercial_charge[profile_id] + \
                                          work_charge[profile_id] + public_charge[profile_id]
    df = pd.DataFrame.from_dict(outside_charge)
    print(f"Saving outside_charge to csv...")
    df.to_csv("outside_charge_2030.csv", index=False)


def describe(f_name):
    df = pd.read_csv(f_name + ".csv")
    print(df.describe())


if __name__ == "__main__":
    files = ["EVC_0_2030", "EVC_1_2030", "EVC_2_2030", "EVC_3_2030",
             "EVP_0_2030", "EVP_1_2030", "EVP_2_2030", "EVP_3_2030", "EVP_4_2030"]
    # create_outside_charge()
    describe("EVP_4_2030_update")



