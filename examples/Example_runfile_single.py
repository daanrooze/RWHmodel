
import RWHmodel

def main():
    testmodel = RWHmodel.Model(
        root = r"C:\Users\example_case",
        name = "example_case",
        mode = "single",
        setup_fn = r"C:\Users\example_case\input\setup_single_run.toml",
        forcing_fn = r"C:\Users\example_case\input\forcing_test.csv",
        demand_fn = r"C:\Users\example_case\input\demand_test.csv",
        demand_transform = False,
        reservoir_initial_state = 0.75,
        #timestep = 86400,
        #t_start = "2003-04-01",
        #t_end = "2005-7-31",
        unit = "mm",
        runoff_source = "model"
        )
    
    testmodel.run()
    
    testmodel.plot(plot_type="meteo", t_start="2000-01-01", t_end="2005-01-01")
    testmodel.plot(plot_type="run", t_start="2000-01-01", t_end="2005-01-01")

    pass

if __name__ == "__main__":
    main()
