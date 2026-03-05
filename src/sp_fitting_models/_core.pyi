def isodesmic_model_direct(x: float, k: float) -> float: ...


def isodesmic_model(conc: float, k: float, num_itr: int) -> float: ...


def temp_isodesmic_model_direct(
	temp: list[float],
	delta_h: float,
	delta_s: float,
	c_tot: float,
	scaler: float,
) -> list[float]: ...


def temp_isodesmic_model(
	temp: list[float],
	delta_h: float,
	delta_s: float,
	c_tot: float,
	scaler: float,
) -> list[float]: ...


def cooperative_model(conc: float, k: float, sigma: float, num_itr: int) -> float: ...


def temp_cooperative_model(
	temp: list[float],
	delta_h: float,
	delta_s: float,
	delta_h_nuc: float,
	c_tot: float,
	scaler: float,
) -> list[float]: ...


def coop_iso_model(conc: float, k_iso: float, k_coop: float, sigma: float, num_itr: int) -> float: ...


def temp_coop_iso_model(
	temp: list[float],
	delta_h_iso: float,
	delta_s_iso: float,
	delta_h_coop: float,
	delta_s_coop: float,
	delta_h_nuc_coop: float,
	c_tot: float,
	scaler: float,
) -> list[float]: ...
