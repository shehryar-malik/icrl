import sys

if __name__ == "__main__":
#    from icrl.constraint_net import main
#    main()
#    exit()
    file_to_run = sys.argv[1]

    # Run specified file. All files must ignore the first argument
    # passed via command line
    if file_to_run == "cpg":
        from icrl.cpg import main
    elif file_to_run == "gail":
        from icrl.gail import main
    elif file_to_run == "airl":
        from icrl.airl import main
    elif file_to_run == "icrl":
        from icrl.icrl import main
    elif file_to_run == "run_policy":
        from icrl.run_policy import main
    elif file_to_run == "random_agent":
        from icrl.random_agent import main
    elif file_to_run == "pruning/train.py":
        from pruning.train import main
    elif file_to_run == "pruning/pruning_env.py":
        from pruning.pruning_env import main
    else:
        raise ValueError("File %s not defined" % file_to_run)

    # Now run
#    from icrl.cpg import main
    main()
