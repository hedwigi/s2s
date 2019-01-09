class Monitor:

    @staticmethod
    def print_params(params):
        print("="*60)
        print("# Model Architecture")
        print("="*60)
        total_num_params = 0
        for param in params:
            num_params = 1
            for dim in param.get_shape():
                num_params *= dim.value
            total_num_params += num_params
            print("    %s, shape : %s, Number of params : %s" % (param.name, str(param.get_shape()), num_params))
        print("-"*60)
        print("# Total model parameters :",total_num_params)
        print("-"*60)
