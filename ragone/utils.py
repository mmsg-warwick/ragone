import pybamm


def get_options(SEI=False, plating=False, lam=False):
    tag = ""
    options = {}
    if SEI:
        tag += "_SEI"
        options["SEI"] = "reaction limited"
        # options["SEI"] = "ec reaction limited"
        options["SEI porosity change"] = "true"
    if plating:
        tag += "_plating"
        options["lithium plating"] = "irreversible"
        options["lithium plating porosity change"] = "true"
    if lam:
        tag += "_lam"
        options["particle mechanics"] = "swelling only"
        options["loss of active material"] = "stress-driven"

    return options, tag


def get_parameter_values(ageing=True):
    parameter_values = pybamm.ParameterValues("OKane2022")
    Chen2020 = pybamm.ParameterValues("Chen2020")
    ORegan2022 = pybamm.ParameterValues("ORegan2022")
    parameter_values["Negative electrode OCP [V]"] = Chen2020[
        "Negative electrode OCP [V]"
    ]

    for param in [
        "Cation transference number",
        "Thermodynamic factor",
        "Electrolyte conductivity [S.m-1]",
        "Electrolyte diffusivity [m2.s-1]",
    ]:
        parameter_values[param] = ORegan2022[param]

    if ageing:
        parameter_values["SEI reaction exchange current density [A.m-2]"] = (
            3.6e-8  # was 3.375e-8
        )
        # 3.5e-8 worked
        parameter_values["SEI kinetic rate constant [m.s-1]"] = 5e-8
        parameter_values["EC diffusivity [m2.s-1]"] = 1e-20
        parameter_values["Lithium plating kinetic rate constant [m.s-1]"] = 4e-12 * 0.8
        parameter_values["Exchange-current density for plating [A.m-2]"] = 8e-4
        # overrides kinetic rate
        parameter_values["Negative electrode LAM constant proportional term [s-1]"] = (
            6.9e-07
        )
        parameter_values["Positive electrode LAM constant proportional term [s-1]"] = (
            6.9e-07
        )
        parameter_values["Negative electrode LAM constant exponential term"] = 1.3
        parameter_values["Positive electrode LAM constant exponential term"] = 1.3
    else:
        parameter_values["SEI kinetic rate constant [m.s-1]"] = 0
        parameter_values["SEI reaction exchange current density [A.m-2]"] = 0
        parameter_values["SEI solvent diffusivity [m2.s-1]"] = 0
        parameter_values["Lithium plating kinetic rate constant [m.s-1]"] = 0
        parameter_values["Negative electrode LAM constant proportional term [s-1]"] = 0
        parameter_values["Positive electrode LAM constant proportional term [s-1]"] = 0

    return parameter_values


def get_var_pts():
    var_pts = {
        # "x_n": 50,
        "x_n": 100,
        "x_s": 30,
        "x_p": 50,
        # "r_n": 20,
        # "r_p": 20,
        "r_n": 30,
        "r_p": 30,
    }
    return var_pts
