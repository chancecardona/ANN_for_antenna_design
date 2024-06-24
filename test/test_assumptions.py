import numpy as np

def test_assumptions(vf_series):
    # Sort vf's by order returned
    vf_by_order = {}
    for vf in vf_series:
        vf_order = vf.get_model_order(vf.poles)
        vf_by_order.setdefault(vf_order, []).append(vf)

    # Verify that each order has same dimensions.
    for order,vf_set in vf_by_order.items():
        prev_data = None
        for vf in vf_set:
            data = [order, len(vf.poles), np.sum(vf.poles.imag == 0)]
            if prev_data is not None:
                if prev_data != data:
                    print("ASSUMPTIONS VIOLATED at order", order)
                    return 1
            prev_data = data

    print("Assumptions verified!")
