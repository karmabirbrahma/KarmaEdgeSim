package edu.boun.edgecloudsim.applications.deepLearning;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

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
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;

import java.util.*;
import java.io.*;
import org.cloudbus.cloudsim.core.CloudSim;
import edu.boun.edgecloudsim.core.SimManager;
import edu.boun.edgecloudsim.mobility.MobilityModel;
import edu.boun.edgecloudsim.utils.Location;

public class DeepEdgeOrchestrator extends EdgeOrchestrator {



    public static final double MAX_DATA_SIZE=2500;

    private int numberOfHost; //used by load balancer
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

    // ==================== S-HEO MOBILITY PRE-RANKING (YOUR NOVELTY) ====================
    private static final boolean MOBILITY_PRERANKING = false;   // false = same as DeepEdge baseline
    private static final int HISTORY_WINDOW = 3;
    private Map<Integer, List<Location>> userPositionHistory = new HashMap<>();

    // ==================== OFFLINE DNN (S-HEO) ====================
    private MultiLayerNetwork offlineDNN = null;

    // ==================== OFFLINE DNN TRAINING DATA COLLECTION ====================
    private static final String DATA_CSV_PATH = "dnn_training_data.csv";
    private PrintWriter dataWriter;
    private boolean isFirstWrite = true;

    public DeepEdgeOrchestrator (String _policy, String _simScenario) {
        super(_policy, _simScenario);
    }

    /*
    @Override
    public void initialize() {
        numberOfHost=SimSettings.getInstance().getNumOfEdgeHosts();

        try {
            fis1 = FIS.createFromString(FCL_definition.fclDefinition1, false);
            fis2 = FIS.createFromString(FCL_definition.fclDefinition2, false);
            fis3 = FIS.createFromString(FCL_definition.fclDefinition3, false);
        } catch (RecognitionException e) {
            SimLogger.printLine("Cannot generate FIS! Terminating simulation...");
            e.printStackTrace();
            System.exit(0);
        }

        if (policy.equals("DDQN")){

            try {
                final String absolutePath = "TheModel";
                m_agent = MultiLayerNetwork.load(new File(absolutePath), false);
            }catch (IOException e) {
                e.printStackTrace();
            }


        }
    }
    */

    //====== DNN policy ====== REALLLLLLLLLL
    @Override
    public void initialize() {
        numberOfHost = SimSettings.getInstance().getNumOfEdgeHosts();

        // 1. Initialize Fuzzy Inference Systems (Unchanged)
        try {
            fis1 = FIS.createFromString(FCL_definition.fclDefinition1, false);
            fis2 = FIS.createFromString(FCL_definition.fclDefinition2, false);
            fis3 = FIS.createFromString(FCL_definition.fclDefinition3, false);
        } catch (RecognitionException e) {
            SimLogger.printLine("Cannot generate FIS!");
            e.printStackTrace();
            System.exit(0);
        }

        // 2. Initialize Neural Networks
        if (policy.equals("DDQN")) {
            try {
                final String absolutePath = "TheModel";
                m_agent = MultiLayerNetwork.load(new File(absolutePath), false);

                // --- Architecture ---
                org.deeplearning4j.nn.conf.MultiLayerConfiguration conf = new org.deeplearning4j.nn.conf.NeuralNetConfiguration.Builder()
                    .seed(123)
                    .list()
                    .layer(0, new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder().nIn(7).nOut(64).activation(org.nd4j.linalg.activations.Activation.RELU).build())
                    .layer(1, new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder().nIn(64).nOut(32).activation(org.nd4j.linalg.activations.Activation.RELU).build())
                    .layer(2, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder().nIn(32).nOut(2).activation(org.nd4j.linalg.activations.Activation.IDENTITY)
                        .lossFunction(org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction.MSE).build())
                    .build();

                offlineDNN = new MultiLayerNetwork(conf);
                offlineDNN.init();

                // --- Manual CSV Injection ---
                String path = "models/models/"; // Adjust if you moved files to models/
                if (!new File(path + "param_0.csv").exists()) path = "models/";
                
                SimLogger.printLine("😊 Loading CSV weights from: " + path);

                offlineDNN.getLayer(0).setParam("W", loadCSV(path + "param_0.csv", 7, 64));
                offlineDNN.getLayer(0).setParam("b", loadCSV(path + "param_1.csv", 1, 64));
                offlineDNN.getLayer(1).setParam("W", loadCSV(path + "param_2.csv", 64, 32));
                offlineDNN.getLayer(1).setParam("b", loadCSV(path + "param_3.csv", 1, 32));
                offlineDNN.getLayer(2).setParam("W", loadCSV(path + "param_4.csv", 32, 2));
                offlineDNN.getLayer(2).setParam("b", loadCSV(path + "param_5.csv", 1, 2));

                SimLogger.printLine("✅ S-HEO Offline DNN Loaded Successfully!");

            } catch (Exception e) {
                SimLogger.printLine("❌ Failed to initialize models!");
                e.printStackTrace();
            }
        }
    }

    // Helper Method to read flat CSV into NDArray
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
    /*
     * (non-Javadoc)
     * @see edu.boun.edgecloudsim.edge_orchestrator.EdgeOrchestrator#getDeviceToOffload(edu.boun.edgecloudsim.edge_client.Task)
     *
     * It is assumed that the edge orchestrator app is running on the edge devices in a distributed manner
     */
    // @Override
    // public int getDeviceToOffload(Task task) {
    //     int result = 0;

    //     if(simScenario.equals("SINGLE_TIER")){
    //         result = SimSettings.GENERIC_EDGE_DEVICE_ID;
    //     }
    //     else if(simScenario.equals("TWO_TIER_WITH_EO")){

    //         if(policy.equals("DDQN")){
    //             counter++;
    //             if (counter > EPISODE_SIZE){
    //                 Task dummyTask = new Task(0, 0, 0, 0, 128, 128, new UtilizationModelFull(), new UtilizationModelFull(), new UtilizationModelFull());
    //                 double wanDelay = SimManager.getInstance().getNetworkModel().getUploadDelay(task.getMobileDeviceId(),
    //                         SimSettings.CLOUD_DATACENTER_ID, dummyTask /* 1 Mbit */);
    //                 double wanBW = (wanDelay == 0) ? 0 : (1 / wanDelay); /* Mbps */
    //                 if(wanBW > 6)
    //                     result = SimSettings.CLOUD_DATACENTER_ID;
    //                 else
    //                     result = SimSettings.GENERIC_EDGE_DEVICE_ID;
    //             }else{
    //                 // ========================================================================
    //                 // S-HEO HYBRID LOGIC INTEGRATION
    //                 // ========================================================================
                    
    //                 // 1. Gather Candidate Servers (All Edge Hosts + Cloud)
    //                 List<Integer> candidateServers = new ArrayList<>();
    //                 for(int i=0; i<numberOfHost; i++) {
    //                     candidateServers.add(i);
    //                 }
    //                 candidateServers.add(SimSettings.CLOUD_DATACENTER_ID); // Cloud is usually ID 14 or similar
                    
    //                 // 2. MOBILITY PRE-RANKING
    //                 // This uses your custom history vector to rank servers based on user movement
    //                 List<Integer> rankedServers = preRankServersByMobility(task.getMobileDeviceId(), candidateServers);
                    
    //                 // Take the top 3 servers to reduce signaling overhead
    //                 int topK = Math.min(3, rankedServers.size());
    //                 List<Integer> topServers = rankedServers.subList(0, topK);
                    
    //                 // 3. OFFLINE DNN PREDICTION
    //                 double bestPredictedDelay = Double.MAX_VALUE;
    //                 int bestServerId = SimSettings.CLOUD_DATACENTER_ID; // Fallback to cloud
                    
    //                 if (offlineDNN != null) {
    //                     for (int serverId : topServers) {
    //                         try {
    //                             // Extract the 7 features for the DNN
    //                             double alpha = task.getCloudletLength(); // Using length as a proxy for alpha/cycles
    //                             double beta = task.getCloudletFileSize(); // Using file size as a proxy for beta/dataSize
    //                             double requiredCycles = task.getCloudletLength();
    //                             double dataSize = task.getCloudletFileSize() + task.getCloudletOutputSize();
                                
    //                             // Estimated network stats (replaces real-time signaling overhead)
    //                             double queueLength = 0.0; // Estimate or get from local cache
    //                             double channelRate = 1.0; // Estimate or get from local cache
                                
    //                             double[] rawFeatures = new double[] { 
    //                                 alpha, beta, requiredCycles, dataSize, serverId, queueLength, channelRate 
    //                             };
                                
    //                             INDArray inputVector = org.nd4j.linalg.factory.Nd4j.create(rawFeatures).reshape(1, 7);
    //                             INDArray prediction = offlineDNN.output(inputVector);
                                
    //                             double predictedDelay = prediction.getDouble(0);
    //                             double predictedEnergy = prediction.getDouble(1);
                                
    //                             // SimLogger.printLine("DNN predicts Server " + serverId + " -> Delay: " + predictedDelay);
                                
    //                             // Find the server with the lowest predicted delay
    //                             if (predictedDelay < bestPredictedDelay) {
    //                                 bestPredictedDelay = predictedDelay;
    //                                 bestServerId = serverId;
    //                             }
    //                         } catch (Exception e) {
    //                             SimLogger.printLine("⚠️ DNN Prediction failed for server " + serverId);
    //                         }
    //                     }
    //                 }

    //                 // 4. FINAL DECISION
    //                 // S-HEO decides to use the DNN's best server. 
    //                 // (Alternatively, you can pass this bestServerId to the DDQN agent as a state feature)
    //                 result = bestServerId;
                    
    //                 // Logging for training/metrics
    //                 logTrainingData(task, result, bestPredictedDelay, 0.0, 0.0, 0.0);

    //                 if (result == SimSettings.CLOUD_DATACENTER_ID){
    //                     numberOfWanOffloadedTask++;
    //                 }
    //                 else if(task.getSubmittedLocation().getServingWlanId() == result){
    //                     numberOfWlanOffloadedTask++;
    //                 }
    //                 else{
    //                     numberOfManOffloadedTask++;
    //                 }
    //                 // ========================================================================
    //             }

    //         }
    //             // ... [Keep your existing FUZZY and NETWORK_BASED else-if blocks exactly as they are below this] ...
    //             else if(policy.equals("FUZZY_COMPETITOR")){
    //                 double utilization = edgeUtilization;
    //                 double cpuSpeed = (double)100 - utilization;
    //                 double videoExecution = SimSettings.getInstance().getTaskLookUpTable()[task.getTaskType()][12];
    //                 double dataSize = task.getCloudletFileSize() + task.getCloudletOutputSize();
    //                 double normalizedDataSize = Math.min(MAX_DATA_SIZE, dataSize)/MAX_DATA_SIZE;

    //                 // Set inputs
    //                 fis3.setVariable("wan_bw", wanBW);
    //                 fis3.setVariable("cpu_speed", cpuSpeed);
    //                 fis3.setVariable("video_execution", videoExecution);
    //                 fis3.setVariable("data_size", normalizedDataSize);

    //                 // Evaluate
    //                 fis3.evaluate();

    //                 /*
    //                 SimLogger.printLine("########################################");
    //                 SimLogger.printLine("wan bw: " + wanBW);
    //                 SimLogger.printLine("cpu_speed: " + cpuSpeed);
    //                 SimLogger.printLine("video_execution: " + videoExecution);
    //                 SimLogger.printLine("data_size: " + normalizedDataSize);
    //                 SimLogger.printLine("offload_decision: " + fis2.getVariable("offload_decision").getValue());
    //                 SimLogger.printLine("########################################");
    //                 */

    //                 if(fis3.getVariable("offload_decision").getValue() > 50)
    //                     result = SimSettings.CLOUD_DATACENTER_ID;
    //                 else
    //                     result = SimSettings.GENERIC_EDGE_DEVICE_ID;
    //             }


    //             else if(policy.equals("NETWORK_BASED")){
    //                 if(wanBW > 6)
    //                     result = SimSettings.CLOUD_DATACENTER_ID;
    //                 else
    //                     result = SimSettings.GENERIC_EDGE_DEVICE_ID;
    //             }
    //             else if(policy.equals("UTILIZATION_BASED")){
    //                 double utilization = edgeUtilization;
    //                 if(utilization > 80)
    //                     result = SimSettings.CLOUD_DATACENTER_ID;
    //                 else
    //                     result = SimSettings.GENERIC_EDGE_DEVICE_ID;
    //             }
    //             else if(policy.equals("HYBRID")){
    //                 double utilization = edgeUtilization;
    //                 if(wanBW > 6 && utilization > 80)
    //                     result = SimSettings.CLOUD_DATACENTER_ID;
    //                 else
    //                     result = SimSettings.GENERIC_EDGE_DEVICE_ID;
    //             }
    //             else {
    //                 SimLogger.printLine("Unknow edge orchestrator policy! Terminating simulation...");
    //                 System.exit(0);
    //             }
    //         //}

    //         }
    //     else {
    //         SimLogger.printLine("Unknow simulation scenario! Terminating simulation...");
    //         System.exit(0);
    //     }

    //     return result;
    // }

    private boolean simScene = false;
    private boolean hasPrintedPolicy = false;
    private boolean ultiSHO = false;
    private boolean isFuzzy = false;
    //===========S-HEO LOGIC HERE=============
    @Override
    public int getDeviceToOffload(Task task) {
        int result = 0;

        if(simScenario.equals("SINGLE_TIER")){
            result = SimSettings.GENERIC_EDGE_DEVICE_ID;
        }
        else if(simScenario.equals("TWO_TIER_WITH_EO")){
            if(!simScene){
                SimLogger.printLine("\n🚀 DEBUG: Checking for Sim Check");
                simScene = true;
            }
            // --- GLOBALS FOR ALL POLICIES TO PREVENT COMPILATION ERRORS ---
            double edgeUtilization = SimManager.getInstance().getEdgeServerManager().getAvgUtilization();
            Task dummyTask = new Task(0, 0, 0, 0, 128, 128, new UtilizationModelFull(), new UtilizationModelFull(), new UtilizationModelFull());
            double wanDelay = SimManager.getInstance().getNetworkModel().getUploadDelay(task.getMobileDeviceId(),
                    SimSettings.CLOUD_DATACENTER_ID, dummyTask /* 1 Mbit */);
            double wanBW = (wanDelay == 0) ? 0 : (1 / wanDelay); /* Mbps */
            // --------------------------------------------------------------

            if(policy.equals("DDQN")){

                if (!hasPrintedPolicy) {
                    SimLogger.printLine("\n🚀 DEBUG: Stoping for Policy check");
                    hasPrintedPolicy = true; 
                }

                counter++;
                if (counter > EPISODE_SIZE){
                    if(wanBW > 6)
                        result = SimSettings.CLOUD_DATACENTER_ID;
                    else
                        result = SimSettings.GENERIC_EDGE_DEVICE_ID;
                }else{
                    // ========================================================================
                    // S-HEO TRUE HYBRID LOGIC (Mobility + DNN + DDQN)
                    // ========================================================================
                    if(!ultiSHO){
                        SimLogger.printLine("\n🚀 DEBUG: Reach here means SHO");
                        ultiSHO = true;
                    }
                    // 1. MOBILITY PRE-RANKING
                    List<Integer> candidateServers = new ArrayList<>();
                    for(int i=0; i<numberOfHost; i++) candidateServers.add(i);
                    candidateServers.add(SimSettings.CLOUD_DATACENTER_ID);
                    
                    List<Integer> rankedServers = preRankServersByMobility(task.getMobileDeviceId(), candidateServers);
                    int topK = Math.min(3, rankedServers.size());
                    List<Integer> topServers = rankedServers.subList(0, topK);
                    
                    // 2. OFFLINE DNN PREDICTION (Find the "Safe" Fallback)
                    double bestPredictedDelay = Double.MAX_VALUE;
                    int dnnBestServer = SimSettings.CLOUD_DATACENTER_ID; 
                    
                    if (offlineDNN != null) {
                        for (int serverId : topServers) {
                            try {
                                double alpha = task.getCloudletLength();
                                double beta = task.getCloudletFileSize();
                                double requiredCycles = task.getCloudletLength();
                                double dataSize = task.getCloudletFileSize() + task.getCloudletOutputSize();
                                double queueLength = 0.0; // Local estimate
                                double channelRate = 1.0; // Local estimate
                                
                                double[] rawFeatures = new double[] { alpha, beta, requiredCycles, dataSize, serverId, queueLength, channelRate };
                                INDArray inputVector = org.nd4j.linalg.factory.Nd4j.create(rawFeatures).reshape(1, 7);
                                INDArray prediction = offlineDNN.output(inputVector);
                                
                                double predictedDelay = prediction.getDouble(0);
                                if (predictedDelay < bestPredictedDelay) {
                                    bestPredictedDelay = predictedDelay;
                                    dnnBestServer = serverId;
                                }
                            } catch (Exception e) {
                                // Silent catch to prevent console spam
                            }
                        }
                    }

                    // 3. ONLINE DDQN PROPOSAL
                    DeepEdgeState currentState = GetFeaturesForAgent(task);
                    INDArray output = m_agent.output(currentState.getState());
                    int ddqnProposedServer = output.argMax().getInt();

                    // 4. S-HEO DECISION ENFORCEMENT
                    // If DDQN's choice is in our top 3 mobility-safe list, we allow it.
                    // If not, we override with the DNN's recommended server to prevent task failure.
                    if (topServers.contains(ddqnProposedServer) || ddqnProposedServer == SimSettings.CLOUD_DATACENTER_ID) {
                        result = ddqnProposedServer;
                    } else {
                        result = dnnBestServer;
                    }
                    
                    // Logging
                    logTrainingData(task, result, bestPredictedDelay, 0.0, 0.0, 0.0);

                    // Standard DeepEdge Counters
                    if (result == SimSettings.CLOUD_DATACENTER_ID){
                        numberOfWanOffloadedTask++;
                    }
                    else if(task.getSubmittedLocation().getServingWlanId() == result){
                        numberOfWlanOffloadedTask++;
                    }
                    else{
                        numberOfManOffloadedTask++;
                    }
                    // ========================================================================
                }

            }
            // ==================== COMPETITOR POLICIES ====================
            else if(policy.equals("FUZZY_COMPETITOR")){
                if (!isFuzzy) {
                    SimLogger.printLine("\n🚀 DEBUG: FUZZY as well running");
                    isFuzzy = true; 
                }
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

                if(fis3.getVariable("offload_decision").getValue() > 50)
                    result = SimSettings.CLOUD_DATACENTER_ID;
                else
                    result = SimSettings.GENERIC_EDGE_DEVICE_ID;
            }
            else if(policy.equals("NETWORK_BASED")){
                if(wanBW > 6) result = SimSettings.CLOUD_DATACENTER_ID;
                else result = SimSettings.GENERIC_EDGE_DEVICE_ID;
            }
            else if(policy.equals("UTILIZATION_BASED")){
                if(edgeUtilization > 80) result = SimSettings.CLOUD_DATACENTER_ID;
                else result = SimSettings.GENERIC_EDGE_DEVICE_ID;
            }
            else if(policy.equals("HYBRID")){ // Note: This is the old baseline Hybrid, not S-HEO
                if(wanBW > 6 && edgeUtilization > 80) result = SimSettings.CLOUD_DATACENTER_ID;
                else result = SimSettings.GENERIC_EDGE_DEVICE_ID;
            }
            else {
                SimLogger.printLine("Unknown edge orchestrator policy! Terminating simulation...");
                System.exit(0);
            }
        }
        else {
            SimLogger.printLine("Unknown simulation scenario! Terminating simulation...");
            System.exit(0);
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

        /*
        if (selectedVM == null){
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

         */


        return selectedVM;
    }

    @Override
    public void processEvent(SimEvent arg0) {
        // Nothing to do!
    }

    @Override
    public void shutdownEntity() {
        // Nothing to do!
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

    // ==================== OFFLINE DNN DATA LOGGING ====================
    private void logTrainingData(Task task, int chosenServerId, 
                                 double predictedDelay, double predictedEnergy,
                                 double actualDelay, double actualEnergy) {
        try {
            if (isFirstWrite) {
                dataWriter = new PrintWriter(new FileWriter(DATA_CSV_PATH));
                dataWriter.println("taskId,alpha,beta,requiredCycles,dataSize,serverId," +
                                   "queueLength,channelRate,predictedDelay,predictedEnergy," +
                                   "actualDelay,actualEnergy");
                isFirstWrite = false;
            }

            StringBuilder sb = new StringBuilder();
            sb.append(task.getCloudletId()).append(",");
            sb.append(task.getCloudletLength()).append(",");
            sb.append(task.getCloudletFileSize()).append(",");
            sb.append(task.getCloudletLength()).append(",");
            sb.append(task.getCloudletFileSize()).append(",");
            sb.append(chosenServerId).append(",");
            sb.append("0.0").append(",");      // queueLength placeholder
            sb.append("1.0").append(",");      // channelRate placeholder
            sb.append(predictedDelay).append(",");
            sb.append(predictedEnergy).append(",");
            sb.append(actualDelay).append(",");
            sb.append(actualEnergy);

            dataWriter.println(sb.toString());
            dataWriter.flush();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}
