reset_vectorstore:
	find src/database -mindepth 1 -not -name '.gitkeep' -exec rm -rf {} +

reset_output:
	find src/output -mindepth 1 -not -name '.gitkeep' -exec rm -rf {} +

reset: reset_database reset_output