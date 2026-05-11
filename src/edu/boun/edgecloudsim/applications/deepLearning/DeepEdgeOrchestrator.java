package edu.boun.edgecloudsim.applications.deepLearning;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import edu.boun.edgecloudsim.applications.sample_app4.FCL_definition;
import org.antlr.runtime.RecognitionException;
import org.cloudbus.cloudsim.Host;
import org.cloudbus.cloudsim.UtilizationModelFull;
import org.cloudbus.cloudsim.Vm;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.core.SimEvent;

import net.sourceforge.jFuzzyLogic.FIS;
import edu.boun.edgecloudsim.cloud_server.CloudVM;
import edu.boun.edgecloudsim.core.SimManager;
import edu.boun.edgecloudsim.core.SimSettings;
import edu.boun.edgecloudsim.edge_orchestrator.EdgeOrchestrator;
import edu.boun.edgecloudsim.edge_server.EdgeHost;
import edu.boun.edgecloudsim.edge_server.EdgeVM;
import edu.boun.edgecloudsim.edge_client.CpuUtilizationModel_Custom;
import edu.boun.edgecloudsim.edge_client.Task;
import edu.boun.edgecloudsim.utils.SimLogger;
import edu.boun.edgecloudsim.mobility.MobilityModel;
import edu.boun.edgecloudsim.utils.Location;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

public class DeepEdgeOrchestrator extends EdgeOrchestrator {

    public static final double MAX_DATA_SIZE = 2500;

    private int numberOfHost; 
    private FIS fis1 = null;
    private FIS fis2 = null;
    private FIS fis3 = null;

    private MultiLayerNetwork m_agent = null;
    private double numberOfWlanOffloadedTask = 0;
    private double numberOfManOffloadedTask = 0;
    private double numberOfWanOffloadedTask = 0;
    private double activeManTaskCount = 0;
    private double activeWanTaskCount = 0;
    private double totalSizeOfActiveManTasks = 0;
    private int counter = 0;
    private final int EPISODE_SIZE = 75000;

    // ==================== S-HEO MOBILITY PRE-RANKING ====================
    private static final boolean MOBILITY_PRERANKING = true;
    private static final int HISTORY_WINDOW = 3;
    private Map<Integer, List<Location>> userPositionHistory = new HashMap<>();

    // ==================== OFFLINE DNN (S-HEO) ====================
    private MultiLayerNetwork offlineDNN = null;

    // ==================== IN-MEMORY DATA COLLECTION ====================
    private class TaskRecord {
        Task task;
        double[] features;
        double predictedDelay;
        int chosenServer;

        public TaskRecord(Task t, double[] f, double pDelay, int server) {
            this.task = t;
            this.features = f;
            this.predictedDelay = pDelay;
            this.chosenServer = server;
        }
    }
    private List<TaskRecord> taskLog = new ArrayList<>();

    public DeepEdgeOrchestrator (String _policy, String _simScenario) {
        super(_policy, _simScenario);
    }

    @Override
    public void initialize() {
        numberOfHost = SimSettings.getInstance().getNumOfEdgeHosts();

        try {
            fis1 = FIS.createFromString(FCL_definition.fclDefinition1, false);
            fis2 = FIS.createFromString(FCL_definition.fclDefinition2, false);
            fis3 = FIS.createFromString(FCL_definition.fclDefinition3, false);
        } catch (RecognitionException e) {
            SimLogger.printLine("Cannot generate FIS!");
            System.exit(0);
        }

        if (policy.equals("DDQN")) {
            try {
                // Load Online DDQN
                final String absolutePath = "TheModel";
                m_agent = MultiLayerNetwork.load(new File(absolutePath), false);

                // Build 5-to-1 Offline DNN Architecture
                org.deeplearning4j.nn.conf.MultiLayerConfiguration conf = new org.deeplearning4j.nn.conf.NeuralNetConfiguration.Builder()
                    .seed(123)
                    .list()
                    .layer(0, new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder().nIn(5).nOut(64).activation(org.nd4j.linalg.activations.Activation.RELU).build())
                    .layer(1, new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder().nIn(64).nOut(32).activation(org.nd4j.linalg.activations.Activation.RELU).build())
                    .layer(2, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder().nIn(32).nOut(1).activation(org.nd4j.linalg.activations.Activation.IDENTITY)
                        .lossFunction(org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction.MSE).build())
                    .build();

                offlineDNN = new MultiLayerNetwork(conf);
                offlineDNN.init();

                String path = "models/models/"; 
                if (!new File(path + "param_0.csv").exists()) path = "models/";
                
                SimLogger.printLine("😊 Loading 5-Feature CSV weights from: " + path);

                offlineDNN.getLayer(0).setParam("W", loadCSV(path + "param_0.csv", 5, 64));
                offlineDNN.getLayer(0).setParam("b", loadCSV(path + "param_1.csv", 1, 64));
                offlineDNN.getLayer(1).setParam("W", loadCSV(path + "param_2.csv", 64, 32));
                offlineDNN.getLayer(1).setParam("b", loadCSV(path + "param_3.csv", 1, 32));
                offlineDNN.getLayer(2).setParam("W", loadCSV(path + "param_4.csv", 32, 1));
                offlineDNN.getLayer(2).setParam("b", loadCSV(path + "param_5.csv", 1, 1));

                SimLogger.printLine("✅ S-HEO Offline DNN Loaded Successfully!");

            } catch (Exception e) {
                SimLogger.printLine("❌ Failed to initialize models! Will run baseline only.");
            }
        }
    }

    private org.nd4j.linalg.api.ndarray.INDArray loadCSV(String file, int rows, int cols) throws Exception {
        double[] data = new double[rows * cols];
        java.util.Scanner sc = new java.util.Scanner(new File(file));
        int i = 0;
        while (sc.hasNextDouble() && i < data.length) {
            data[i++] = sc.nextDouble();
        }
        sc.close();
        return org.nd4j.linalg.factory.Nd4j.create(data).reshape(rows, cols);
    } 

    @Override
    public int getDeviceToOffload(Task task) {
        int result = 0;

        if(simScenario.equals("SINGLE_TIER")){
            result = SimSettings.GENERIC_EDGE_DEVICE_ID;
        }
        else if(simScenario.equals("TWO_TIER_WITH_EO")){

            double edgeUtilization = SimManager.getInstance().getEdgeServerManager().getAvgUtilization();
            Task dummyTask = new Task(0, 0, 0, 0, 128, 128, new UtilizationModelFull(), new UtilizationModelFull(), new UtilizationModelFull());
            double wanDelay = SimManager.getInstance().getNetworkModel().getUploadDelay(task.getMobileDeviceId(),
                    SimSettings.CLOUD_DATACENTER_ID, dummyTask);
            double wanBW = (wanDelay == 0) ? 0 : (1 / wanDelay); 

            if(policy.equals("DDQN")){
                counter++;
                if (counter > EPISODE_SIZE){
                    if(wanBW > 6) result = SimSettings.CLOUD_DATACENTER_ID;
                    else result = SimSettings.GENERIC_EDGE_DEVICE_ID;
                } else {
                    List<Integer> candidateServers = new ArrayList<>();
                    for(int i=0; i<numberOfHost; i++) candidateServers.add(i);
                    candidateServers.add(SimSettings.CLOUD_DATACENTER_ID);
                    
                    List<Integer> rankedServers = preRankServersByMobility(task.getMobileDeviceId(), candidateServers);
                    int topK = Math.min(3, rankedServers.size());
                    List<Integer> topServers = rankedServers.subList(0, topK);
                    
                    double bestPredictedDelay = Double.MAX_VALUE;
                    int dnnBestServer = SimSettings.CLOUD_DATACENTER_ID; 
                    
                    // NEW 5 NON-REDUNDANT FEATURES
                    double requiredCycles = task.getCloudletLength();
                    double dataSize = task.getCloudletFileSize() + task.getCloudletOutputSize();

                    if (offlineDNN != null) {
                        for (int serverId : topServers) {
                            try {
                                double[] rawFeatures = new double[] { requiredCycles, dataSize, serverId, edgeUtilization, wanBW };
                                INDArray inputVector = org.nd4j.linalg.factory.Nd4j.create(rawFeatures).reshape(1, 5);
                                INDArray prediction = offlineDNN.output(inputVector);
                                
                                double predictedDelay = prediction.getDouble(0); 
                                if (predictedDelay < bestPredictedDelay) {
                                    bestPredictedDelay = predictedDelay;
                                    dnnBestServer = serverId;
                                }
                            } catch (Exception e) {}
                        }
                    }

                    DeepEdgeState currentState = GetFeaturesForAgent(task);
                    INDArray output = m_agent.output(currentState.getState());
                    int ddqnProposedServer = output.argMax().getInt();

                    if (topServers.contains(ddqnProposedServer) || ddqnProposedServer == SimSettings.CLOUD_DATACENTER_ID) {
                        result = ddqnProposedServer;
                    } else {
                        result = dnnBestServer;
                    }
                    
                    // Log to Memory
                    double[] featuresToSave = new double[] { requiredCycles, dataSize, result, edgeUtilization, wanBW };
                    taskLog.add(new TaskRecord(task, featuresToSave, bestPredictedDelay, result));

                    if (result == SimSettings.CLOUD_DATACENTER_ID) numberOfWanOffloadedTask++;
                    else if(task.getSubmittedLocation().getServingWlanId() == result) numberOfWlanOffloadedTask++;
                    else numberOfManOffloadedTask++;
                }

            }
            else if(policy.equals("NETWORK_BASED")){
                if(wanBW > 6) result = SimSettings.CLOUD_DATACENTER_ID;
                else result = SimSettings.GENERIC_EDGE_DEVICE_ID;

                // Log Baseline Ground Truth to Memory
                double requiredCycles = task.getCloudletLength();
                double dataSize = task.getCloudletFileSize() + task.getCloudletOutputSize();
                double[] featuresToSave = new double[] { requiredCycles, dataSize, result, edgeUtilization, wanBW };
                
                taskLog.add(new TaskRecord(task, featuresToSave, 0.0, result));
            }
            else if(policy.equals("FUZZY_COMPETITOR")){
                double utilization = edgeUtilization;
                double cpuSpeed = (double)100 - utilization;
                double videoExecution = SimSettings.getInstance().getTaskLookUpTable()[task.getTaskType()][12];
                double dataSize = task.getCloudletFileSize() + task.getCloudletOutputSize();
                double normalizedDataSize = Math.min(MAX_DATA_SIZE, dataSize)/MAX_DATA_SIZE;

                fis3.setVariable("wan_bw", wanBW);
                fis3.setVariable("cpu_speed", cpuSpeed);
                fis3.setVariable("video_execution", videoExecution);
                fis3.setVariable("data_size", normalizedDataSize);
                fis3.evaluate();

                if(fis3.getVariable("offload_decision").getValue() > 50) result = SimSettings.CLOUD_DATACENTER_ID;
                else result = SimSettings.GENERIC_EDGE_DEVICE_ID;
            }
            else if(policy.equals("UTILIZATION_BASED")){
                if(edgeUtilization > 80) result = SimSettings.CLOUD_DATACENTER_ID;
                else result = SimSettings.GENERIC_EDGE_DEVICE_ID;
            }
            else if(policy.equals("HYBRID")){
                if(wanBW > 6 && edgeUtilization > 80) result = SimSettings.CLOUD_DATACENTER_ID;
                else result = SimSettings.GENERIC_EDGE_DEVICE_ID;
            }
        }
        return result;
    }


    public DeepEdgeState GetFeaturesForAgent(Task task){
        Task dummyTask = new Task(0, 0, 0, 0, 128, 128, new UtilizationModelFull(), new UtilizationModelFull(), new UtilizationModelFull());

        DeepEdgeState currentState = new DeepEdgeState();
        ArrayList<Double> edgeCapacities = new ArrayList<>();

        int numberOfHost = SimSettings.getInstance().getNumOfEdgeHosts();

        double wanDelay = SimManager.getInstance().getNetworkModel().getUploadDelay(task.getMobileDeviceId(),
                SimSettings.CLOUD_DATACENTER_ID, dummyTask /* 1 Mbit */);

        double wanBW = (wanDelay == 0) ? 0 : (1 / wanDelay); /* Mbps */

        currentState.setWanBw(wanBW/20.21873);

        double manDelayF = SimManager.getInstance().getNetworkModel().getUploadDelayForTraining(SimSettings.GENERIC_EDGE_DEVICE_ID,
                SimSettings.GENERIC_EDGE_DEVICE_ID, dummyTask );

        double manBW = (manDelayF == 0) ? 0 : (1 / manDelayF);


        double manDelay = getManDelayForAgent();
        currentState.setManDelay(manDelay);

        double taskRequiredCapacity = ((CpuUtilizationModel_Custom)task.getUtilizationModelCpu()).predictUtilization(SimSettings.VM_TYPES.EDGE_VM);
        currentState.setTaskReqCapacity(taskRequiredCapacity/800);

        int wlanID = task.getSubmittedLocation().getServingWlanId();
        currentState.setWlanID((double)wlanID / (numberOfHost - 1));

        int nearestEdgeHostId = 0;


        for(int hostIndex=0; hostIndex<numberOfHost; hostIndex++){
            List<EdgeVM> vmArray = SimManager.getInstance().getEdgeServerManager().getVmList(hostIndex);
            EdgeHost host = (EdgeHost)(vmArray.get(0).getHost()); //all VMs have the same host

            double totalUtilizationForEdgeServer=0;
            for(int vmIndex=0; vmIndex<vmArray.size(); vmIndex++){
                totalUtilizationForEdgeServer += vmArray.get(vmIndex).getCloudletScheduler().getTotalUtilizationOfCpu(CloudSim.clock());
            }

            double totalCapacity = 100 * vmArray.size();
            double averageCapacity = (totalCapacity - totalUtilizationForEdgeServer)  / vmArray.size();
            double normalizedCapacity = averageCapacity / 100;

            if (normalizedCapacity < 0){
                normalizedCapacity = 0;
            }
            edgeCapacities.add(normalizedCapacity);

            if (host.getLocation().getServingWlanId() == task.getSubmittedLocation().getServingWlanId()){
                nearestEdgeHostId = hostIndex;
            }

        }

        currentState.setAvailVmInEdge(edgeCapacities);
        currentState.setNearestEdgeHostId((double)nearestEdgeHostId / numberOfHost);

        double delay_sensitivity = SimSettings.getInstance().getTaskLookUpTable()[task.getTaskType()][12];

        currentState.setDelaySensitivity(delay_sensitivity);

        currentState.setNumberOfWlanOffloadedTask(numberOfWlanOffloadedTask/EPISODE_SIZE);
        currentState.setNumberOfManOffloadedTask(numberOfManOffloadedTask/EPISODE_SIZE);
        currentState.setNumberOfWanOffloadedTask(numberOfWanOffloadedTask/EPISODE_SIZE);
        currentState.setActiveManTaskCount(activeManTaskCount/25);
        currentState.setActiveWanTaskCount(activeWanTaskCount/25);


        return currentState;

    }

    public double getManDelayForAgent(){
        double delay = 0;
        double mu = 0;
        double lambda = 0;
        double bandwidth = 1300*1024; //Kbps , C

        if (totalSizeOfActiveManTasks == 0){
            mu = bandwidth;
        }else{
            mu = bandwidth / (totalSizeOfActiveManTasks * 8);
        }

        lambda = activeManTaskCount;


        if (lambda >= mu){
            return 0;
        }else{
            delay = 1 / (mu - lambda);
            return delay;
        }
    }


    @Override
    public Vm getVmToOffload(Task task, int deviceId) {
        Vm selectedVM = null;

        if(deviceId == SimSettings.CLOUD_DATACENTER_ID){
            //Select VM on cloud devices via Least Loaded algorithm!
            double selectedVmCapacity = 0; //start with min value
            List<Host> list = SimManager.getInstance().getCloudServerManager().getDatacenter().getHostList();
            for (int hostIndex=0; hostIndex < list.size(); hostIndex++) {
                List<CloudVM> vmArray = SimManager.getInstance().getCloudServerManager().getVmList(hostIndex);
                for(int vmIndex=0; vmIndex<vmArray.size(); vmIndex++){
                    double requiredCapacity = ((CpuUtilizationModel_Custom)task.getUtilizationModelCpu()).predictUtilization(vmArray.get(vmIndex).getVmType());
                    double targetVmCapacity = (double)100 - vmArray.get(vmIndex).getCloudletScheduler().getTotalUtilizationOfCpu(CloudSim.clock());
                    if(requiredCapacity <= targetVmCapacity && targetVmCapacity > selectedVmCapacity){
                        selectedVM = vmArray.get(vmIndex);
                        selectedVmCapacity = targetVmCapacity;
                    }
                }
            }
        }
        else if(deviceId == SimSettings.GENERIC_EDGE_DEVICE_ID){
            //Select VM on edge devices via Least Loaded algorithm!
            double selectedVmCapacity = 0; //start with min value
            for(int hostIndex=0; hostIndex<numberOfHost; hostIndex++){
                List<EdgeVM> vmArray = SimManager.getInstance().getEdgeServerManager().getVmList(hostIndex);
                for(int vmIndex=0; vmIndex<vmArray.size(); vmIndex++){
                    double requiredCapacity = ((CpuUtilizationModel_Custom)task.getUtilizationModelCpu()).predictUtilization(vmArray.get(vmIndex).getVmType());
                    double targetVmCapacity = (double)100 - vmArray.get(vmIndex).getCloudletScheduler().getTotalUtilizationOfCpu(CloudSim.clock());
                    if(requiredCapacity <= targetVmCapacity && targetVmCapacity > selectedVmCapacity){
                        selectedVM = vmArray.get(vmIndex);
                        selectedVmCapacity = targetVmCapacity;
                    }
                }
            }
        }
        else{
            //if the host is specifically defined!
            List<EdgeVM> vmArray = SimManager.getInstance().getEdgeServerManager().getVmList(deviceId);

            //Select VM on edge devices via Least Loaded algorithm!
            double selectedVmCapacity = 0; //start with min value
            for(int vmIndex=0; vmIndex<vmArray.size(); vmIndex++){
                double requiredCapacity = ((CpuUtilizationModel_Custom)task.getUtilizationModelCpu()).predictUtilization(vmArray.get(vmIndex).getVmType());
                double targetVmCapacity = (double)100 - vmArray.get(vmIndex).getCloudletScheduler().getTotalUtilizationOfCpu(CloudSim.clock());
                if(requiredCapacity <= targetVmCapacity && targetVmCapacity > selectedVmCapacity){
                    selectedVM = vmArray.get(vmIndex);
                    selectedVmCapacity = targetVmCapacity;
                }
            }
        }
        return selectedVM;
    }

    @Override
    public void processEvent(SimEvent arg0) {
        // Nothing to do!
    }

    @Override
    public void shutdownEntity() {
        SimLogger.printLine("💾 Saving actual DNN training data to CSV...");
        try {
            PrintWriter writer = new PrintWriter(new FileWriter("real_dnn_training_data.csv"));
            // 5 Features + Predict + Actual
            writer.println("taskId,requiredCycles,dataSize,serverId,edgeUtilization,wanBW,predictedDelay,actualDelay");

            for (TaskRecord record : taskLog) {
                if (record.task.getCloudletStatus() == org.cloudbus.cloudsim.Cloudlet.SUCCESS) {
                    double actualDelay = record.task.getFinishTime() - record.task.getSubmissionTime();
                    
                    StringBuilder sb = new StringBuilder();
                    sb.append(record.task.getCloudletId()).append(",");
                    sb.append(record.features[0]).append(","); // requiredCycles
                    sb.append(record.features[1]).append(","); // dataSize
                    sb.append(record.chosenServer).append(",");
                    sb.append(record.features[3]).append(","); // edgeUtilization
                    sb.append(record.features[4]).append(","); // wanBW
                    sb.append(record.predictedDelay).append(","); 
                    sb.append(actualDelay);

                    writer.println(sb.toString());
                }
            }
            writer.flush();
            writer.close();
            SimLogger.printLine("✅ Real dataset saved as real_dnn_training_data.csv!");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public void startEntity() {
        // Nothing to do!
    }

    private List<Integer> preRankServersByMobility(int deviceId, List<Integer> candidateServers) {
        if (!MOBILITY_PRERANKING || candidateServers.isEmpty()) {
            return candidateServers;   // flag OFF = original DeepEdge behavior
        }

        // Minimal version: only direction bonus (compiles and works)
        MobilityModel mobilityModel = SimManager.getInstance().getMobilityModel();
        Location currentLoc = mobilityModel.getLocation(deviceId, CloudSim.clock());

        List<Location> history = userPositionHistory.getOrDefault(deviceId, new ArrayList<>());
        history.add(currentLoc);
        if (history.size() > HISTORY_WINDOW) history.remove(0);
        userPositionHistory.put(deviceId, history);

        if (history.size() < 2) return candidateServers;

        Location prev = history.get(history.size() - 2);
        double currentX = currentLoc.getXPos();
        double currentY = currentLoc.getYPos();
        double prevX = prev.getXPos();
        double prevY = prev.getYPos();

        double dx = currentX - prevX;
        double dy = currentY - prevY;

        List<ServerScore> scored = new ArrayList<>();
        for (int serverId : candidateServers) {
            double score = 0.0;
            if (dx != 0 || dy != 0) {
                double dot = dx * (0 - currentX) + dy * (0 - currentY);
                score = (dot > 0) ? 0.3 : -0.2;
            }
            scored.add(new ServerScore(serverId, score));
        }

        scored.sort((a, b) -> Double.compare(a.score, b.score));
        List<Integer> ranked = new ArrayList<>();
        for (ServerScore s : scored) ranked.add(s.serverId);
        return ranked;
    }

    private static class ServerScore {
        int serverId;
        double score;
        ServerScore(int id, double s) { serverId = id; score = s; }
    }

}
