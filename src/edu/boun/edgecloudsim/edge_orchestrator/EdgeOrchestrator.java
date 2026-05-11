package edu.boun.edgecloudsim.edge_orchestrator;

import org.cloudbus.cloudsim.Vm;
import org.cloudbus.cloudsim.core.SimEntity;

import edu.boun.edgecloudsim.edge_client.Task;

public abstract class EdgeOrchestrator extends SimEntity{
	protected String policy;
	protected String simScenario;
	
	public EdgeOrchestrator(String _policy, String _simScenario){
		super("EdgeOrchestrator");
		policy = _policy;
		simScenario = _simScenario;
	}
	
	/*
	 * initialize edge orchestrator if needed
	 */
	public abstract void initialize();
	
	/*
	 * decides where to offload
	 */
	public abstract int getDeviceToOffload(Task task);
	
	/*
	 * returns proper VM from the edge orchestrator point of view
	 */
	public abstract Vm getVmToOffload(Task task, int deviceId);
}
