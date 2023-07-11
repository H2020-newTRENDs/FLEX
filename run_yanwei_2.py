from run_yanwei_main import run_region


if __name__ == "__main__":
    regions = [
        # 'SK', 'SE', 'PL', 'MT', 'BE', 'EE', 'EL', 'LV', 'IT',
        'CZ', 'RO', 'PT', 'SI', 'HR', 'HU', 'NL', 'BG', 'AT',
        # 'DE', 'DK', 'FI', 'LU', 'FR', 'ES', 'IE', 'LT', 'CY'
    ]
    for index, region in enumerate(regions):
        print(f'{region} --> {index + 1}/{len(regions)}')
        run_region(region)







