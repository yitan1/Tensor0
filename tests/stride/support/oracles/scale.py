"""Reference values for selected scaling."""


def reference_scale(base, factor):
    return base.at[..., 2:50:3].set(base[..., 2:50:3] * factor[..., None])
