from irc.manager import AgentManager

if __name__=='__main__':
    manager = AgentManager(
        'store', defaults='irc-defaults.yaml', eval_interval=1, disp_interval=1,
    )

    config = {
        'env_param': [1, 0.001, 0.1, 0, 0.0286, 0.2, 0.0476, 0.2, 0.0667, 0.2],
    }
    config = manager.get_config(config)
    manager.process(config, num_epochs=18)
