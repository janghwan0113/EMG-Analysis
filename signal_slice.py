import numpy as np

def kinematic_crop(data_to_crop, timetuple):
    cropped_emg_lift = []
    cropped_emg_lower = []

    for data_row in data_to_crop:
        temp_lift_row = []
        temp_lower_row = []

        for lift_start, lift_end, lower_start, lower_end in zip(timetuple[0], timetuple[1], timetuple[2], timetuple[3]):
            temp_lift_row.append(data_row[lift_start:lift_end])
            temp_lower_row.append(data_row[lower_start:lower_end])

        cropped_emg_lift.append(temp_lift_row)
        cropped_emg_lower.append(temp_lower_row)

    return np.array(cropped_emg_lift, dtype='object'), np.array(cropped_emg_lower, dtype='object')
