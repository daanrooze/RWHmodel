
import RWHmodel

def main():
    testmodel = RWHmodel.Model(
        root = r"C:\Users\example_case",
        name = "example_case",
        setup_fn = r"C:\Users\example_case\input\setup_batch_run.toml",
        forcing_fn = r"C:\Users\example_case\input\forcing_test.csv",
        demand_fn = [0.5, 5, 10],
        demand_transform = False,
        reservoir_range = [10, 200, 10],
        reservoir_initial_state = 0.75,
        #timestep = 86400,
        #t_start = "2003-04-01",
        #t_end = "2005-7-31",
        unit = "mm"
        )
    
    testmodel.batch_run(method="consecutive_timesteps", log=True, save=False)
    
    testmodel.plot(plot_type="run_coverage")
    testmodel.plot(plot_type="system_curve", validation=True, T_return_list=[1,2,5,10,20])
    testmodel.plot(plot_type="saving_curve", T_return_list=[1,2,5,10])

    pass

if __name__ == "__main__":
    main()
