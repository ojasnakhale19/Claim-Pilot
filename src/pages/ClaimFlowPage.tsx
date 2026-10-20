import { useState, useCallback, useEffect, useRef } from 'react';
import {
  Camera,
  CheckCircle,
  MessageSquare,
  RefreshCw,
  Shield,
  Zap,
  AlertTriangle,
  Check,
  X,
  Brain,
  ChevronDown,
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  CinematicBackground,
  CSSWheel,
  InspectionUpload,
  FourViewInference,
  DamageAnnotation,
  FuturisticChat,
  TransitionWrapper,
  StageIndicator,
  LoadingOverlay,
} from '../components/cinematic';
import { cn, isValidImageFile, isValidVideoFile, createObjectURL, revokeObjectURL, getFileSizeString, generateId, formatNumber, assessFourView, assessHybrid, type DamageDetection } from '../lib/utils';
import { type Detection } from '../lib/yoloInference';
import { type DamageDetection as PricingDamageDetection, calculateSettlement, type ModelPricing, loadCarPricingData } from '../lib/settlementCalculator';
import { generateAINegotiationResponse } from '../lib/aiService';
import { Card, CardContent, CardFooter } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { Progress } from '../components/ui/Progress';
import { Button } from '../components/ui/Button';

const MAX_PHOTOS = 8;

type Severity = 'minor' | 'moderate' | 'severe' | 'uncertain';

interface AssessmentData {
  component: string;
  damageType: string;
  severity: Severity;
  costMin: number;
  costMax: number;
  repairApproach: string;
  explanation: string;
  settlementOffer: number;
  settlementMin: number;
  settlementMax: number;
}

const MOCK_ASSESSMENTS: Record<Severity, AssessmentData> = {
  minor: {
    component: 'Front Bumper',
    damageType: 'Surface scratches & minor deformation',
    severity: 'minor',
    costMin: 4000,
    costMax: 6000,
    repairApproach: 'Repair and repaint',
    explanation: 'The visible damage appears consistent with a repair and repaint. The final repair cost may vary after a physical inspection.',
    settlementOffer: 3800,
    settlementMin: 3200,
    settlementMax: 4500,
  },
  moderate: {
    component: 'Side Mirror (Right)',
    damageType: 'Housing damage & cracked glass',
    severity: 'moderate',
    costMin: 8000,
    costMax: 12000,
    repairApproach: 'Replace component',
    explanation: 'The mirror housing shows significant damage with cracked glass. Replacement is recommended for safety. Labour costs include calibration.',
    settlementOffer: 7500,
    settlementMin: 6200,
    settlementMax: 9500,
  },
  severe: {
    component: 'Front Bumper & Fender',
    damageType: 'Structural deformation & paint damage',
    severity: 'severe',
    costMin: 25000,
    costMax: 35000,
    repairApproach: 'Replace components & structural repair',
    explanation: 'The damage extends to structural elements requiring panel replacement and alignment. A physical inspection is mandatory before final estimate.',
    settlementOffer: 22000,
    settlementMin: 18000,
    settlementMax: 28000,
  },
  uncertain: {
    component: 'Unable to determine',
    damageType: 'Insufficient visual evidence',
    severity: 'uncertain',
    costMin: 0,
    costMax: 0,
    repairApproach: 'Requires human review',
    explanation: 'We couldn\'t confidently assess the damage from these photos. Try adding a clearer close-up or a photo from another angle.',
    settlementOffer: 0,
    settlementMin: 0,
    settlementMax: 0,
  },
};

interface UploadedPhoto {
  id: string;
  file: File;
  preview: string;
  name: string;
  size: number;
  type: 'image' | 'video';
}

type WorkflowStage = 'upload' | 'assessing' | 'assessment' | 'chat' | 'confirmed' | 'humanReview';
type AssessmentMode = 'demo' | 'ai';

interface ChatMessage {
  id: string;
  sender: 'bot' | 'user';
  text: string;
  timestamp: Date;
  isOffer?: boolean;
  offerAmount?: number;
}

interface AssessmentResult {
  type: Severity;
  data: AssessmentData;
}

const workflowStages = [
  { id: 'upload', label: 'UPLOAD', icon: <Camera className="h-4 w-4" /> },
  { id: 'assessing', label: 'SCAN', icon: <Zap className="h-4 w-4" /> },
  { id: 'assessment', label: 'REPORT', icon: <AlertTriangle className="h-4 w-4" /> },
  { id: 'chat', label: 'NEGOTIATE', icon: <MessageSquare className="h-4 w-4" /> },
  { id: 'confirmed', label: 'DONE', icon: <CheckCircle className="h-4 w-4" /> },
];

export function ClaimFlowPage() {
  // Core state
  const [photos, setPhotos] = useState<UploadedPhoto[]>([]);
  const [stage, setStage] = useState<WorkflowStage>('upload');
  const [previousStage, setPreviousStage] = useState<WorkflowStage | null>(null);
  const [assessment, setAssessment] = useState<AssessmentResult | null>(null);
  const [assessmentProgress, setAssessmentProgress] = useState(0);
  const [assessmentStep, setAssessmentStep] = useState(0);

  // Vehicle info state
  const [vehicleInfo, setVehicleInfo] = useState({
    company: '',
    model: '',
    year: '',
    variant: '',
    causeOfDamage: '',
  });

  // Car data state
  const [carData, setCarData] = useState<Record<string, string[]>>({});
  const [showCompanyDropdown, setShowCompanyDropdown] = useState(false);
  const [showModelDropdown, setShowModelDropdown] = useState(false);
  const [companyFilter, setCompanyFilter] = useState('');
  const [modelFilter, setModelFilter] = useState('');
  const companyDropdownRef = useRef<HTMLDivElement>(null);
  const modelDropdownRef = useRef<HTMLDivElement>(null);

  // Pricing data state
  const [pricingData, setPricingData] = useState<ModelPricing[]>([]);

  // Four-view upload state (for hybrid assessment)
  const [viewStates] = useState<Record<string, { file: File | null; preview: string | null }>>({
    top: { file: null, preview: null },
    bottom: { file: null, preview: null },
    left: { file: null, preview: null },
    right: { file: null, preview: null },
  });
  
  // Ref to store the latest files from FourViewInference
  const fourViewFilesRef = useRef<Record<string, File | null>>({
    top: null,
    bottom: null,
    left: null,
    right: null,
  });

  const REQUIRED_VIEWS = ['top', 'bottom', 'left', 'right'] as const;

  // Fetch car data on mount
  useEffect(() => {
    fetch('/car_data.json')
      .then(res => res.json())
      .then(data => setCarData(data))
      .catch(console.error);
  }, []);

  // Fetch pricing data on mount
  useEffect(() => {
    loadCarPricingData().then(data => setPricingData(data));
  }, []);

  // Close dropdowns when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (companyDropdownRef.current && !companyDropdownRef.current.contains(event.target as Node)) {
        setShowCompanyDropdown(false);
      }
      if (modelDropdownRef.current && !modelDropdownRef.current.contains(event.target as Node)) {
        setShowModelDropdown(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);
  
  // Assessment mode
  const [assessmentMode, setAssessmentMode] = useState<AssessmentMode>('demo');
  
  // Chat state
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [isBotTyping, setIsBotTyping] = useState(false);
  const [chatStage, setChatStage] = useState<'initial' | 'negotiating' | 'confirmed'>('initial');
  const [currentOffer, setCurrentOffer] = useState(0);
  const [offerAccepted, setOfferAccepted] = useState(false);
  
  // UI state
  const [photoInputKey, setPhotoInputKey] = useState(0);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [showLightbox, setShowLightbox] = useState<UploadedPhoto | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Assessment steps for simulated processing
  const assessmentSteps = [
    'Scanning vehicle surfaces…',
    'Identifying damage patterns…',
    'Calculating repair estimates…',
  ];

  // Cleanup object URLs on unmount
  useEffect(() => {
    return () => {
      photos.forEach(p => revokeObjectURL(p.preview));
    };
  }, [photos]);

  // Scroll to bottom of chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Photo handling
  const processPhotos = useCallback((files: File[]) => {
    const validPhotos: UploadedPhoto[] = [];
    const invalidFiles: string[] = [];

    files.forEach(file => {
      const isImage = isValidImageFile(file);
      const isVideo = isValidVideoFile(file);
      
      if (isImage || isVideo) {
        validPhotos.push({
          id: generateId('photo'),
          file,
          preview: createObjectURL(file),
          name: file.name,
          size: file.size,
          type: isVideo ? 'video' : 'image',
        });
      } else {
        invalidFiles.push(file.name);
      }
    });

    if (validPhotos.length > 0) {
      setPhotos(prev => [...prev, ...validPhotos].slice(0, MAX_PHOTOS));
    }

    if (invalidFiles.length > 0) {
      setErrors(prev => ({
        ...prev,
        photos: `Unsupported files: ${invalidFiles.join(', ')}. Use JPG, PNG, WebP (max 10MB) or MP4, MOV (max 50MB).`
      }));
      setTimeout(() => setErrors(prev => { const n = {...prev}; delete n.photos; return n; }), 5000);
    }
  }, []);

  const removePhoto = useCallback((id: string) => {
    const photo = photos.find(p => p.id === id);
    if (photo) revokeObjectURL(photo.preview);
    setPhotos(prev => prev.filter(p => p.id !== id));
  }, [photos]);

  const openLightbox = useCallback((photo: UploadedPhoto) => {
    setShowLightbox(photo);
  }, []);

  const handleAIAssessmentError = useCallback((error: string) => {
    setErrors(prev => ({ ...prev, assessment: error }));
    setTimeout(() => setErrors(prev => { const n = {...prev}; delete n.assessment; return n; }), 8000);
  }, []);

  const handleAIValidationError = useCallback((validation: any) => {
    // Validation errors are handled inline by FourViewUpload
    console.log('Validation error:', validation);
  }, []);

  // Hybrid AI Assessment handler (local YOLO + cloud multimodal reasoning)
  const handleHybridAssessmentComplete = useCallback(async (localResult: {
    views: Record<string, {
      detections: Detection[];
      annotatedImage: string | null;
      processingTime: number;
    }>;
    totalDetections: number;
    primaryComponent: string;
    damageSummary: string;
    severity: 'minor' | 'moderate' | 'severe' | 'uncertain';
    estimatedRepairCostMin: number;
    estimatedRepairCostMax: number;
    settlementOffer: number;
    settlementMin: number;
    settlementMax: number;
    reviewRequired: boolean;
    reviewFlags: Array<{ code: string; view: string; message: string }>;
    limitations: string[];
  }) => {
    // First, get the four-view assessment from backend to get validated detections
    try {
      setAssessmentProgress(80);
      setAssessmentStep(3);
      
      // Convert local detections to backend format
      const fourViewRequest = {
        top: viewStates.top.file!,
        bottom: viewStates.bottom.file!,
        left: viewStates.left.file!,
        right: viewStates.right.file!,
        confidence_threshold: 0.25,
        image_size: 640,
      };
      
      // Call backend four-view assessment to get validated detections
      const backendResult = await assessFourView(fourViewRequest);
      
      // Now call hybrid AI with backend detections and images
      setAssessmentProgress(90);
      setAssessmentStep(4);
      
      // Convert files to base64
      const viewsBase64: Record<string, string> = {};
      for (const view of REQUIRED_VIEWS) {
        const file = fourViewFilesRef.current[view]!;
        viewsBase64[view] = await fileToBase64(file);
      }
      
      // Convert detections to hybrid format
      const detections: Record<string, DamageDetection[]> = {};
      for (const view of REQUIRED_VIEWS) {
        detections[view] = backendResult.views[view].detections;
      }
      
      const hybridResult = await assessHybrid({
        vehicle: {
          make: vehicleInfo.company,
          model: vehicleInfo.model,
          year: vehicleInfo.year ? parseInt(vehicleInfo.year) : undefined,
          variant: vehicleInfo.variant,
        },
        views: viewsBase64,
        detections,
        confidence_threshold: 0.25,
        image_size: 640,
      });
      
      // Build assessment data from hybrid result
      const data: AssessmentData = {
        component: hybridResult.primary_component,
        damageType: hybridResult.damage_summary,
        severity: hybridResult.severity,
        costMin: hybridResult.estimated_repair_cost_min,
        costMax: hybridResult.estimated_repair_cost_max,
        repairApproach: 'Hybrid AI (YOLO + Multimodal)',
        explanation: `${hybridResult.overall_reasoning} AI Confidence: ${(hybridResult.confidence * 100).toFixed(0)}%. ${hybridResult.damage_items.length} damage items identified with multimodal verification.`,
        settlementOffer: hybridResult.settlement_offer,
        settlementMin: hybridResult.settlement_min,
        settlementMax: hybridResult.settlement_max,
      };
      
      setAssessment({ type: hybridResult.severity, data });
      setPreviousStage('assessing');
      setStage('assessment');
    } catch (error: any) {
      console.warn('Hybrid assessment failed, using local result:', error);
      // Fallback to local result
      const data: AssessmentData = {
        component: localResult.primaryComponent,
        damageType: localResult.damageSummary,
        severity: localResult.severity,
        costMin: localResult.estimatedRepairCostMin,
        costMax: localResult.estimatedRepairCostMax,
        repairApproach: 'Local AI detection (hybrid unavailable)',
        explanation: `Local AI assessment detected ${localResult.totalDetections} damage instance(s). ${localResult.damageSummary}. Settlement calculated using local inference.`,
        settlementOffer: localResult.settlementOffer,
        settlementMin: localResult.settlementMin,
        settlementMax: localResult.settlementMax,
      };
      
      setAssessment({ type: localResult.severity, data });
      setPreviousStage('assessing');
      setStage('assessment');
    }
  }, [vehicleInfo]);

  // Helper to convert file to base64
  const fileToBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const result = reader.result as string;
        // Remove data URL prefix
        const base64 = result.split(',')[1];
        resolve(base64);
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  };

  // Demo assessment logic (original) - now uses database pricing
  const startDemoAssessment = useCallback(async () => {
    if (photos.length === 0) {
      setErrors(prev => ({ ...prev, photos: 'Please add at least one photo' }));
      return;
    }

    setPreviousStage(stage);
    setStage('assessing');
    setAssessmentProgress(0);
    setAssessmentStep(0);

    for (let i = 0; i < assessmentSteps.length; i++) {
      await new Promise(r => setTimeout(r, 1200));
      setAssessmentStep(i + 1);
      setAssessmentProgress(Math.round(((i + 1) / assessmentSteps.length) * 100));
    }

    await new Promise(r => setTimeout(r, 600));
    
    // Determine assessment type based on number of photos (demo logic)
    let assessmentType: Severity;
    if (photos.length >= 4) assessmentType = 'minor';
    else if (photos.length === 3) assessmentType = 'moderate';
    else if (photos.length === 2) assessmentType = 'severe';
    else assessmentType = 'uncertain';

    // Use database pricing if vehicle info and pricing data available
    if (vehicleInfo.company && vehicleInfo.model && pricingData.length > 0) {
      // Create mock detections based on assessment type
      const mockDetections: PricingDamageDetection[] = assessmentType === 'minor' ? [
        { part: 'front bumper', severity: 'minor', confidence: 0.9 },
      ] : assessmentType === 'moderate' ? [
        { part: 'side mirror', severity: 'moderate', confidence: 0.85 },
        { part: 'front bumper', severity: 'minor', confidence: 0.8 },
      ] : assessmentType === 'severe' ? [
        { part: 'front bumper', severity: 'severe', confidence: 0.95 },
        { part: 'fender', severity: 'severe', confidence: 0.9 },
        { part: 'headlight', severity: 'moderate', confidence: 0.85 },
      ] : [];

      const settlement = calculateSettlement(pricingData, {
        manufacturer: vehicleInfo.company,
        model: vehicleInfo.model,
        causeOfDamage: vehicleInfo.causeOfDamage,
      }, mockDetections);

      const data: AssessmentData = {
        component: mockDetections[0]?.part || 'Front Bumper',
        damageType: mockDetections.map(d => `${d.severity} ${d.part}`).join(', '),
        severity: assessmentType,
        costMin: settlement.totalRepairMin + settlement.totalReplaceMin,
        costMax: settlement.totalRepairMax + settlement.totalReplaceMax,
        repairApproach: settlement.recommendedApproach === 'repair' ? 'Repair' : 
                        settlement.recommendedApproach === 'replace' ? 'Replace' : 'Mixed repair & replace',
        explanation: `Demo assessment based on ${photos.length} photo(s). ${mockDetections.map(d => `${d.severity} ${d.part}`).join(', ')}. Settlement calculated using ${vehicleInfo.company} ${vehicleInfo.model} parts database.`,
        settlementOffer: settlement.settlementOffer,
        settlementMin: settlement.settlementMin,
        settlementMax: settlement.settlementMax,
      };
      
      setAssessment({ type: assessmentType, data });
    } else {
      // Fallback to original mock data
      const result = MOCK_ASSESSMENTS[assessmentType];
      setAssessment({ type: assessmentType, data: result });
    }
    
    setPreviousStage('assessing');
    setStage('assessment');
  }, [photos, stage, pricingData, vehicleInfo]);

  // Unified start assessment
  const startAssessment = useCallback(async () => {
    if (assessmentMode === 'ai') {
      // FourViewUpload handles the assessment internally
      // This is triggered by the FourViewUpload component's button
      return;
    }
    await startDemoAssessment();
  }, [assessmentMode, startDemoAssessment]);

  // Chat logic
  const addMessage = useCallback((message: Omit<ChatMessage, 'id' | 'timestamp'>) => {
    setMessages(prev => [...prev, {
      ...message,
      id: generateId('msg'),
      timestamp: new Date(),
    }]);
  }, []);

  const handleSendMessage = useCallback(async () => {
    if (!chatInput.trim() || isBotTyping) return;
    
    const userText = chatInput.trim();
    setChatInput('');
    addMessage({ sender: 'user', text: userText });
    setIsBotTyping(true);

    const assessmentData = assessment?.data;
    if (!assessmentData) {
      setIsBotTyping(false);
      return;
    }

    const lowerText = userText.toLowerCase();

    // Call hybrid AI service (Gemini / OpenAI / Intelligent Fallback)
    const aiRes = await generateAINegotiationResponse({
      vehicle: vehicleInfo,
      damageSummary: assessmentData.damageType,
      severity: assessmentData.severity,
      primaryComponent: assessmentData.component,
      costMin: assessmentData.costMin,
      costMax: assessmentData.costMax,
      settlementOffer: currentOffer || assessmentData.settlementOffer,
      settlementMin: assessmentData.settlementMin,
      settlementMax: assessmentData.settlementMax,
      userMessage: userText,
    });

    setIsBotTyping(false);

    addMessage({
      sender: 'bot',
      text: aiRes.message,
      isOffer: aiRes.isOffer,
      offerAmount: aiRes.newOffer,
    });

    if (aiRes.newOffer) {
      setCurrentOffer(aiRes.newOffer);
      if (lowerText.includes('accept') || lowerText.includes('agree') || lowerText.includes('confirm')) {
        setOfferAccepted(true);
        setChatStage('confirmed');
      } else {
        setChatStage('negotiating');
      }
    }
  }, [chatInput, assessment, currentOffer, addMessage, vehicleInfo]);

  const handleConfirmOffer = useCallback(() => {
    if (!assessment) return;
    const finalAmount = currentOffer || assessment.data.settlementOffer;
    addMessage({ 
      sender: 'bot', 
      text: `Your demo settlement is confirmed at ₹${formatNumber(finalAmount)}. Claim reference: CP-${generateId('').slice(0, 8).toUpperCase()}. No real payment has been processed—this is a demonstration only.`,
      isOffer: true,
      offerAmount: finalAmount
    });
    setPreviousStage('chat');
    setStage('confirmed');
  }, [assessment, currentOffer, addMessage]);

  const handleReset = useCallback(() => {
    photos.forEach(p => revokeObjectURL(p.preview));
    setPhotos([]);
    setStage('upload');
    setPreviousStage(null);
    setAssessment(null);
    setMessages([]);
    setChatInput('');
    setChatStage('initial');
    setCurrentOffer(0);
    setOfferAccepted(false);
    setErrors({});
    setShowLightbox(null);
  }, [photos]);

  const handleHumanReview = useCallback(() => {
    setPreviousStage('assessment');
    setStage('humanReview');
    addMessage({
      sender: 'bot',
      text: 'Your request for human review has been logged (demo mode). A claims adjuster will contact you within 2 business days. You can also upload clearer photos in the meantime.',
    });
  }, [addMessage]);

  const handleAddMorePhotos = useCallback(() => {
    setPreviousStage('assessment');
    setStage('upload');
    setAssessment(null);
  }, []);

  const handleEnterChat = useCallback(() => {
    setPreviousStage('assessment');
    setStage('chat');
  }, []);

  // Initial bot message when entering chat
  useEffect(() => {
    if (stage === 'chat' && messages.length === 0 && assessment) {
      const data = assessment.data;
      addMessage({
        sender: 'bot',
        text: `Based on this demo assessment, the estimated garage repair cost is ₹${formatNumber(data.costMin)}–₹${formatNumber(data.costMax)}. We can discuss a cash settlement of ₹${formatNumber(data.settlementOffer)}, subject to applicable policy terms.`,
        isOffer: true,
        offerAmount: data.settlementOffer,
      });
      setCurrentOffer(data.settlementOffer);
    }
  }, [stage, messages.length, assessment, addMessage]);

  // Lightbox modal
  const renderLightbox = () => {
    if (!showLightbox) return null;
    return (
      <motion.div
        className="fixed inset-0 z-50 flex items-center justify-center bg-black/95 backdrop-blur-sm p-4"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        onClick={() => setShowLightbox(null)}
        role="dialog"
        aria-modal="true"
        aria-label="Evidence preview"
      >
        <motion.button
          onClick={() => setShowLightbox(null)}
          className="absolute top-6 right-6 z-10 flex h-12 w-12 items-center justify-center rounded-full bg-graphite-800/80 backdrop-blur text-white hover:bg-graphite-700 transition-colors"
          whileHover={{ scale: 1.1, rotate: 90 }}
          whileTap={{ scale: 0.95 }}
          aria-label="Close preview"
        >
          <X className="h-6 w-6" />
        </motion.button>
        <motion.div
          className="relative max-h-[85vh] max-w-[90vw]"
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          transition={{ duration: 0.3, ease: [0.34, 1.56, 0.64, 1] }}
        >
          <img
            src={showLightbox.preview}
            alt="Damage evidence preview"
            className="rounded-2xl shadow-[0_0_80px_-20px_rgba(0,0,0,0.8)] object-contain"
          />
        </motion.div>
        <motion.div
          className="absolute bottom-6 left-1/2 -translate-x-1/2 text-graphite-400 text-sm font-mono flex items-center gap-2"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <span>{showLightbox.name}</span>
          <span className="text-electric-400/50">·</span>
          <span>{getFileSizeString(showLightbox.size)}</span>
        </motion.div>
      </motion.div>
    );
  };

  return (
    <div className="min-h-screen bg-background relative">
      {/* Cinematic Background */}
      <CinematicBackground isHero={stage === 'upload'} />

      {/* Header */}
      <header className="sticky top-0 z-40 w-full border-b border-graphite-800 bg-background/90 backdrop-blur-xl supports-[backdrop-filter]:bg-background/80">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex h-16 items-center justify-between">
            <div className="flex items-center gap-3">
              <motion.div
                className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-electric-500 to-cyan-500"
                initial={{ scale: 0, rotate: -180 }}
                animate={{ scale: 1, rotate: 0 }}
                transition={{ duration: 0.6, delay: 0.1, ease: [0.34, 1.56, 0.64, 1] }}
              >
                <Zap className="h-5 w-5 text-background" aria-hidden="true" />
              </motion.div>
              <span className="font-bold text-xl text-graphite-100 tracking-tight">ClaimPilot</span>
            </div>
            <div className="flex items-center gap-4">
              <StageIndicator currentStage={stage} stages={workflowStages} />
              <Badge variant="secondary" className="text-[10px] px-2 py-1 font-mono">
                DEMO
              </Badge>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12 relative z-10">
        {/* Introduction / Hero Section - only on upload stage */}
        <AnimatePresence mode="wait">
          {stage === 'upload' && (
            <motion.section
              className="mb-10 relative"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
              aria-labelledby="intro-heading"
            >
              {/* 3D Wheel showcase */}
              <motion.div
                className="absolute right-0 top-1/2 -translate-y-1/2 w-72 h-72 md:w-96 md:h-96 lg:w-[320px] lg:h-[320px] opacity-10 pointer-events-none"
                initial={{ opacity: 0, x: 50 }}
                animate={{ opacity: 0.1, x: 0 }}
                transition={{ duration: 1, delay: 0.5 }}
              >
                <CSSWheel size={320} />
              </motion.div>

              <div className="relative z-10 max-w-2xl">
                <motion.div
                  className="flex items-center gap-2 mb-4"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.5, delay: 0.2 }}
                >
                  <span className="px-3 py-1 rounded-full bg-electric-500/10 border border-electric-500/20 text-electric-400 text-[10px] font-mono tracking-wider">
                    AUTOMATED INSPECTION
                  </span>
                  <span className="text-graphite-500 text-[10px] font-mono">ISO 3779 COMPLIANT</span>
                </motion.div>

                <motion.h1
                  id="intro-heading"
                  className="text-4xl sm:text-5xl lg:text-6xl font-bold tracking-tight text-graphite-100 leading-tight"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: 0.3 }}
                >
                  Vehicle Damage
                  <br />
                  <span className="text-gradient-electric">Automated Assessment</span>
                </motion.h1>

                <motion.p
                  className="mt-5 text-lg text-graphite-400 leading-relaxed max-w-xl"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: 0.4 }}
                >
                  Upload photos of the damage. Our system simulates an automated inspection analysis,
                  providing estimated repair costs and a settlement negotiation workflow.
                </motion.p>

                {/* Vehicle Information Inputs */}
                <motion.div
                  className="mt-8 space-y-4"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: 0.5 }}
                >
                  <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                    {/* Car Company */}
                    <div className="relative">
                      <label 
                        htmlFor="vehicle-company"
                        className="block text-xs font-medium text-graphite-400 mb-2 tracking-wider uppercase"
                      >
                        Car Company
                      </label>
                      <div className="relative" ref={companyDropdownRef}>
                        <input
                          type="text"
                          id="vehicle-company"
                          value={vehicleInfo.company}
                          onChange={(e) => {
                            const value = e.target.value;
                            setCompanyFilter(value);
                            setVehicleInfo(prev => ({ ...prev, company: value, model: '' }));
                            setShowCompanyDropdown(true);
                          }}
                          onFocus={() => setShowCompanyDropdown(true)}
                          onBlur={() => setTimeout(() => setShowCompanyDropdown(false), 200)}
                          placeholder="Select or type..."
                          className={cn(
                            'w-full h-11 px-4 rounded-xl border bg-graphite-900/50 text-graphite-100 placeholder:text-graphite-500',
                            'focus:outline-none focus:ring-2 focus:ring-electric-500 focus:border-transparent',
                            'border-graphite-600 hover:border-electric-500/50',
                            'transition-all'
                          )}
                          autoComplete="off"
                        />
                        <ChevronDown className="absolute right-4 top-1/2 -translate-y-1/2 h-5 w-5 text-graphite-500 pointer-events-none" />
                        
                        {/* Company Dropdown */}
                        {showCompanyDropdown && (
                          <motion.div
                            className="absolute top-full left-0 right-0 mt-1 z-20 rounded-xl border border-graphite-600 bg-graphite-900/95 backdrop-blur-xl shadow-elevated max-h-60 overflow-y-auto"
                            initial={{ opacity: 0, y: -10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -10 }}
                            transition={{ duration: 0.15 }}
                          >
                            <input
                              type="text"
                              value={companyFilter}
                              onChange={(e) => setCompanyFilter(e.target.value)}
                              placeholder="Search companies..."
                              className="p-3 w-full border-b border-graphite-700 bg-graphite-900/50 rounded-t-xl focus:outline-none focus:ring-2 focus:ring-electric-500 text-graphite-100 placeholder:text-graphite-500"
                            />
                            <div className="max-h-[300px] overflow-y-auto">
                              {Object.keys(carData)
                                .filter(m => m.toLowerCase().includes(companyFilter.toLowerCase()))
                                .map((manufacturer) => (
                                  <button
                                    key={manufacturer}
                                    onClick={() => {
                                      setVehicleInfo(prev => ({ ...prev, company: manufacturer, model: '' }));
                                      setShowCompanyDropdown(false);
                                      setCompanyFilter('');
                                    }}
                                    className={cn(
                                      'w-full px-4 py-3 text-left text-graphite-100 hover:bg-electric-500/10 hover:text-electric-400 transition-colors',
                                      vehicleInfo.company === manufacturer && 'bg-electric-500/20 text-electric-400'
                                    )}
                                  >
                                    {manufacturer}
                                  </button>
                                ))}
                              {Object.keys(carData).length === 0 && (
                                <div className="p-4 text-center text-graphite-500 text-sm">Loading...</div>
                              )}
                            </div>
                          </motion.div>
                        )}
                      </div>
                    </div>

                    {/* Car Model */}
                    <div className="relative">
                      <label 
                        htmlFor="vehicle-model"
                        className="block text-xs font-medium text-graphite-400 mb-2 tracking-wider uppercase"
                      >
                        Car Model
                      </label>
                      <div className="relative" ref={modelDropdownRef}>
                        <input
                          type="text"
                          id="vehicle-model"
                          value={vehicleInfo.model}
                          onChange={(e) => {
                            const value = e.target.value;
                            setModelFilter(value);
                            setVehicleInfo(prev => ({ ...prev, model: value }));
                            setShowModelDropdown(true);
                          }}
                          onFocus={() => {
                            if (vehicleInfo.company && carData[vehicleInfo.company]?.length) {
                              setShowModelDropdown(true);
                            }
                          }}
                          onBlur={() => setTimeout(() => setShowModelDropdown(false), 200)}
                          placeholder={vehicleInfo.company ? "Select or type..." : "Select company first"}
                          disabled={!vehicleInfo.company || !carData[vehicleInfo.company]?.length}
                          className={cn(
                            'w-full h-11 px-4 rounded-xl border bg-graphite-900/50 text-graphite-100 placeholder:text-graphite-500',
                            'focus:outline-none focus:ring-2 focus:ring-electric-500 focus:border-transparent',
                            'border-graphite-600 hover:border-electric-500/50',
                            'transition-all',
                            !vehicleInfo.company && 'opacity-50 cursor-not-allowed'
                          )}
                          autoComplete="off"
                        />
                        <ChevronDown className="absolute right-4 top-1/2 -translate-y-1/2 h-5 w-5 text-graphite-500 pointer-events-none" />
                        
                        {/* Model Dropdown */}
                        {showModelDropdown && vehicleInfo.company && carData[vehicleInfo.company]?.length && (
                          <motion.div
                            className="absolute top-full left-0 right-0 mt-1 z-20 rounded-xl border border-graphite-600 bg-graphite-900/95 backdrop-blur-xl shadow-elevated max-h-60 overflow-y-auto"
                            initial={{ opacity: 0, y: -10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -10 }}
                            transition={{ duration: 0.15 }}
                          >
                            <input
                              type="text"
                              value={modelFilter}
                              onChange={(e) => setModelFilter(e.target.value)}
                              placeholder="Search models..."
                              className="p-3 w-full border-b border-graphite-700 bg-graphite-900/50 rounded-t-xl focus:outline-none focus:ring-2 focus:ring-electric-500 text-graphite-100 placeholder:text-graphite-500"
                            />
                            <div className="max-h-[300px] overflow-y-auto">
                              {carData[vehicleInfo.company]
                                .filter(m => m.toLowerCase().includes(modelFilter.toLowerCase()))
                                .map((model) => (
                                  <button
                                    key={model}
                                    onClick={() => {
                                      setVehicleInfo(prev => ({ ...prev, model }));
                                      setShowModelDropdown(false);
                                      setModelFilter('');
                                    }}
                                    className={cn(
                                      'w-full px-4 py-3 text-left text-graphite-100 hover:bg-electric-500/10 hover:text-electric-400 transition-colors',
                                      vehicleInfo.model === model && 'bg-electric-500/20 text-electric-400'
                                    )}
                                  >
                                    {model}
                                  </button>
                                ))}
                            </div>
                          </motion.div>
                        )}
                      </div>
                    </div>

                    {/* Cause of Damage */}
                    <div className="sm:col-span-2 lg:col-span-3">
                      <label 
                        htmlFor="cause-of-damage"
                        className="block text-xs font-medium text-graphite-400 mb-2 tracking-wider uppercase"
                      >
                        Cause of Damage
                      </label>
                      <textarea
                        id="cause-of-damage"
                        value={vehicleInfo.causeOfDamage}
                        onChange={(e) => setVehicleInfo(prev => ({ ...prev, causeOfDamage: e.target.value }))}
                        placeholder="Describe what happened (e.g., rear-end collision, hail damage, parking scrape)..."
                        rows={2}
                        className={cn(
                          'w-full px-4 py-3 rounded-xl border bg-graphite-900/50 text-graphite-100 placeholder:text-graphite-500 resize-none',
                          'focus:outline-none focus:ring-2 focus:ring-electric-500 focus:border-transparent',
                          'border-graphite-600 hover:border-electric-500/50',
                          'transition-all'
                        )}
                      />
                    </div>
                  </div>
                </motion.div>

                <motion.div
                  className="mt-8 flex flex-wrap items-center gap-4"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: 0.5 }}
                >
                  <div className="flex items-center gap-3 px-4 py-2 rounded-xl bg-graphite-800/50 border border-graphite-700">
                    <Shield className="h-5 w-5 text-electric-400" />
                    <span className="text-sm text-graphite-300">Designed for minor motor insurance claims</span>
                  </div>
                  <div className="flex items-center gap-3 px-4 py-2 rounded-xl bg-graphite-800/50 border border-graphite-700">
                    <Zap className="h-5 w-5 text-amber-400" />
                    <span className="text-sm text-graphite-300">Simulated AI assessment</span>
                  </div>
                  <div className="flex items-center gap-3 px-4 py-2 rounded-xl bg-graphite-800/50 border border-graphite-700">
                    <MessageSquare className="h-5 w-5 text-success-400" />
                    <span className="text-sm text-graphite-300">Interactive settlement negotiation</span>
                  </div>
                </motion.div>
                
                {/* Assessment Mode Toggle */}
                <motion.div
                  className="mt-8 p-4 rounded-2xl border border-graphite-700 bg-graphite-900/50"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: 0.6 }}
                >
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-electric-500/20 to-cyan-500/20">
                        <Brain className="h-5 w-5 text-electric-400" />
                      </div>
                      <div>
                        <p className="font-medium text-graphite-100">Assessment Mode</p>
                        <p className="text-xs text-graphite-500">Choose between demo simulation or real AI analysis</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2 bg-graphite-800/50 rounded-xl p-1">
                      <Button
                        variant={assessmentMode === 'demo' ? 'default' : 'ghost'}
                        size="sm"
                        onClick={() => setAssessmentMode('demo')}
                        className="rounded-lg"
                      >
                        Demo Mode
                      </Button>
                      <Button
                        variant={assessmentMode === 'ai' ? 'default' : 'ghost'}
                        size="sm"
                        onClick={() => setAssessmentMode('ai')}
                        className="rounded-lg"
                      >
                        <Brain className="h-3.5 w-3.5 mr-1" />
                        AI Assessment
                      </Button>
                    </div>
                  </div>
                  <p className="text-xs text-graphite-500">
                    {assessmentMode === 'demo' 
                      ? 'Demo mode uses simulated assessments based on photo count. No real AI inference.' 
                      : 'AI mode uses YOLOv11 damage detection on 4 required views (top, bottom, left, right). Requires backend server.'}
                  </p>
                </motion.div>
              </div>
            </motion.section>
          )}
        </AnimatePresence>

        {/* Loading Overlay */}
        <LoadingOverlay
          isVisible={stage === 'assessing'}
          message="AUTOMATED INSPECTION IN PROGRESS"
          subMessage={assessmentSteps[assessmentStep] || assessmentSteps[assessmentSteps.length - 1]}
          progress={assessmentProgress}
        />

        {/* Main Workflow Container */}
        <TransitionWrapper
          stage={stage}
          previousStage={previousStage ?? undefined}
          variant="sweep"
          className="relative"
        >
          <div className="rounded-3xl border border-graphite-700 bg-graphite-900/40 backdrop-blur-2xl overflow-hidden relative">
            {/* Top accent line */}
            <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-transparent via-electric-500/50 to-transparent" aria-hidden="true" />
            
            <div className="p-6 sm:p-8">
              {(() => {
                switch (stage) {
                  case 'upload':
                    if (assessmentMode === 'ai') {
                      return (
                        <FourViewInference
                          onInferenceComplete={handleHybridAssessmentComplete}
                          onError={handleAIAssessmentError}
                          onValidationError={handleAIValidationError}
                          onFilesChange={(files) => {
                            fourViewFilesRef.current = files;
                          }}
                          disabled={stage !== 'upload'}
                        />
                      );
                    }
                    return (
                      <InspectionUpload
                        photos={photos}
                        onAddPhotos={processPhotos}
                        onRemovePhoto={removePhoto}
                        onOpenLightbox={openLightbox}
                        onStartAssessment={startAssessment}
                        maxPhotos={MAX_PHOTOS}
                        errors={errors}
                        photoInputKey={photoInputKey}
                        setPhotoInputKey={setPhotoInputKey}
                      />
                    );
                  case 'assessing':
                    return (
                      <div className="text-center py-12">
                        <motion.div
                          className="relative w-32 h-32 mx-auto mb-6"
                          initial={{ scale: 0 }}
                          animate={{ scale: 1 }}
                          transition={{ duration: 0.5, ease: [0.34, 1.56, 0.64, 1] }}
                        >
                          <Progress 
                            value={assessmentProgress} 
                            className="absolute top-0 left-0 w-full h-full" 
                            max={100}
                          />
                          <motion.div
                            className="absolute inset-0 flex items-center justify-center"
                            animate={{ rotate: 360 }}
                            transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                          >
                            <Zap className="h-12 w-12 text-electric-400" />
                          </motion.div>
                        </motion.div>
                        <motion.h3
                          className="text-lg font-medium text-graphite-100 mb-2"
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                        >
                          Analyzing vehicle damage…
                        </motion.h3>
                        <motion.p
                          className="text-sm text-graphite-400 mb-8"
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: 0.1 }}
                        >
                          {assessmentSteps[assessmentStep] || assessmentSteps[assessmentSteps.length - 1]}
                        </motion.p>
                        <motion.div
                          className="max-w-xs mx-auto space-y-3 text-left"
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          transition={{ delay: 0.2 }}
                        >
                          {assessmentSteps.map((step, index) => (
                            <motion.div
                              key={step}
                              className="flex items-center gap-3 text-sm"
                              initial={{ opacity: 0, x: -20 }}
                              animate={{ opacity: 1, x: 0 }}
                              transition={{ duration: 0.3, delay: 0.3 + index * 0.15 }}
                            >
                              <motion.div
                                className={cn(
                                  'h-7 w-7 rounded-full flex items-center justify-center text-[11px] font-medium flex-shrink-0',
                                  index < assessmentStep ? 'bg-success-500 text-background' :
                                  index === assessmentStep ? 'bg-electric-500 text-background animate-pulse' :
                                  'bg-graphite-700 text-graphite-500'
                                )}
                              >
                                {index < assessmentStep ? <Check className="h-4 w-4" /> : index + 1}
                              </motion.div>
                              <span className={cn(index <= assessmentStep ? 'text-graphite-200' : 'text-graphite-500')}>{step}</span>
                            </motion.div>
                          ))}
                        </motion.div>
                      </div>
                    );
                  case 'assessment':
                    return assessment ? (
                      <DamageAnnotation
                        assessment={assessment}
                        photos={photos}
                        onContinue={handleEnterChat}
                        onAddMorePhotos={handleAddMorePhotos}
                        onHumanReview={handleHumanReview}
                      />
                    ) : null;
                  case 'chat':
                    return assessment ? (
                      <FuturisticChat
                        messages={messages}
                        chatInput={chatInput}
                        setChatInput={setChatInput}
                        isBotTyping={isBotTyping}
                        chatStage={chatStage}
                        offerAccepted={offerAccepted}
                        currentOffer={currentOffer}
                        assessment={assessment}
                        onSendMessage={handleSendMessage}
                        onConfirmOffer={handleConfirmOffer}
                        onIncreaseOffer={() => {
                          const data = assessment!.data;
                          const newOffer = Math.min(data.settlementMax, (currentOffer || data.settlementOffer) + 500);
                          setCurrentOffer(newOffer);
                          addMessage({
                            sender: 'bot',
                            text: `I can revise the offer to ₹${formatNumber(newOffer)}. Does that work for you?`,
                            isOffer: true,
                            offerAmount: newOffer,
                          });
                        }}
                      />
                    ) : null;
                  case 'confirmed':
                    return (
                      <motion.div
                        className="text-center space-y-6"
                        initial={{ opacity: 0, y: 30 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
                      >
                        <motion.div
                          className="flex h-24 w-24 items-center justify-center rounded-full bg-success-500/10 border border-success-500/30 mx-auto"
                          initial={{ scale: 0, rotate: -180 }}
                          animate={{ scale: 1, rotate: 0 }}
                          transition={{ duration: 0.6, delay: 0.2, ease: [0.34, 1.56, 0.64, 1] }}
                        >
                          <CheckCircle className="h-12 w-12 text-success-400" />
                        </motion.div>
                        <motion.div
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: 0.3 }}
                        >
                          <h2 className="text-2xl font-bold text-graphite-100">Demo Settlement Confirmed</h2>
                          <p className="mt-2 text-graphite-400">
                            The illustrative settlement has been agreed. No real payment has been processed.
                          </p>
                        </motion.div>

                        <Card className="border-graphite-700 bg-graphite-900/50 max-w-md mx-auto">
                          <CardContent className="pt-6 pb-6 px-6 space-y-4">
                            <motion.div
                              className="flex items-center justify-center gap-3 text-graphite-500"
                              initial={{ opacity: 0 }}
                              animate={{ opacity: 1 }}
                              transition={{ delay: 0.4 }}
                            >
                              <div className="w-px h-8 bg-graphite-700" />
                              <motion.div
                                className="flex h-10 w-10 items-center justify-center rounded-full bg-electric-500/10 border border-electric-500/20"
                                animate={{ scale: [1, 1.05, 1] }}
                                transition={{ duration: 2, repeat: Infinity }}
                              >
                                <Shield className="h-5 w-5 text-electric-400" />
                              </motion.div>
                              <div className="w-px h-8 bg-graphite-700" />
                            </motion.div>
                            <motion.div
                              className="space-y-3 text-center pt-2 border-t border-graphite-700"
                              initial={{ opacity: 0, y: 10 }}
                              animate={{ opacity: 1, y: 0 }}
                              transition={{ delay: 0.5 }}
                            >
                              <p className="text-xs font-mono text-graphite-400 uppercase tracking-wider">CLAIM REFERENCE</p>
                              <p className="font-mono text-xl font-semibold text-graphite-100">CP-{generateId('').slice(0, 8).toUpperCase()}</p>
                            </motion.div>
                            {assessment && (
                              <motion.div
                                className="space-y-3 text-center pt-4 border-t border-graphite-700"
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: 0.6 }}
                              >
                                <p className="text-xs font-mono text-graphite-400 uppercase tracking-wider">FINAL SETTLEMENT</p>
                                <p className="text-3xl font-bold text-success-400 font-mono">₹{formatNumber(currentOffer || assessment.data.settlementOffer)}</p>
                                <p className="text-sm text-graphite-400">
                                  Assessed: {assessment.data.component} · {assessment.data.severity}
                                </p>
                                <p className="text-sm text-graphite-400">
                                  Est. repair: ₹{formatNumber(assessment.data.costMin)}–₹{formatNumber(assessment.data.costMax)}
                                </p>
                              </motion.div>
                            )}
                            <motion.p
                              className="text-xs text-graphite-500 text-center bg-graphite-800/50 rounded-xl p-3 border border-graphite-700"
                              initial={{ opacity: 0 }}
                              animate={{ opacity: 1 }}
                              transition={{ delay: 0.7 }}
                            >
                              This is a demonstration only. No real settlement, payment, or claim has been created.
                            </motion.p>
                          </CardContent>
                          <CardFooter className="pt-0">
                            <motion.button
                              onClick={handleReset}
                              className="w-full gap-2 px-6 py-3 rounded-xl bg-gradient-to-r from-electric-500 to-cyan-500 text-background font-semibold shadow-glow-electric hover:from-electric-400 hover:to-cyan-400 transition-all"
                              whileHover={{ y: -2, boxShadow: '0 0 30px -5px rgba(14, 165, 233, 0.5)' }}
                              whileTap={{ scale: 0.98 }}
                              initial={{ opacity: 0, y: 10 }}
                              animate={{ opacity: 1, y: 0 }}
                              transition={{ delay: 0.8 }}
                            >
                              <RefreshCw className="h-4 w-4" />
                              Run Another Demo
                            </motion.button>
                          </CardFooter>
                        </Card>
                      </motion.div>
                    );
                  case 'humanReview':
                    return (
                      <motion.div
                        className="text-center space-y-6"
                        initial={{ opacity: 0, y: 30 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
                      >
                        <motion.div
                          className="flex h-24 w-24 items-center justify-center rounded-full bg-amber-500/10 border border-amber-500/30 mx-auto"
                          initial={{ scale: 0, rotate: -180 }}
                          animate={{ scale: 1, rotate: 0 }}
                          transition={{ duration: 0.6, delay: 0.2, ease: [0.34, 1.56, 0.64, 1] }}
                        >
                          <AlertTriangle className="h-12 w-12 text-amber-400" />
                        </motion.div>
                        <motion.div
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: 0.3 }}
                        >
                          <h2 className="text-2xl font-bold text-graphite-100">Human Review Required</h2>
                          <p className="mt-2 text-graphite-400 max-w-md mx-auto">
                            This assessment requires a human adjuster to review the evidence before a settlement can be discussed.
                          </p>
                        </motion.div>

                        <Card className="border-graphite-700 bg-graphite-900/50 max-w-md mx-auto">
                          <CardContent className="pt-6 pb-6 px-6 space-y-4">
                            <motion.div
                              className="space-y-3"
                              initial={{ opacity: 0 }}
                              animate={{ opacity: 1 }}
                              transition={{ delay: 0.4 }}
                            >
                              <motion.button
                                onClick={handleAddMorePhotos}
                                className="w-full gap-2 px-6 py-3 rounded-xl border border-graphite-600 bg-transparent text-graphite-300 hover:border-electric-500 hover:text-electric-400 hover:bg-electric-500/10 transition-all font-medium"
                                whileHover={{ y: -2 }}
                                whileTap={{ scale: 0.98 }}
                              >
                                <Camera className="h-4 w-4" />
                                Add Clearer Photos
                              </motion.button>
                              <motion.button
                                onClick={handleHumanReview}
                                className="w-full gap-2 px-6 py-3 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-400 hover:bg-amber-500/20 hover:border-amber-500 hover:text-amber-300 transition-all font-medium"
                                whileHover={{ y: -2, boxShadow: '0 0 20px -5px rgba(245, 158, 11, 0.3)' }}
                                whileTap={{ scale: 0.98 }}
                              >
                                <MessageSquare className="h-4 w-4" />
                                Request Human Review
                              </motion.button>
                            </motion.div>
                            <motion.p
                              className="text-xs text-graphite-500 text-center"
                              initial={{ opacity: 0 }}
                              animate={{ opacity: 1 }}
                              transition={{ delay: 0.5 }}
                            >
                              In demo mode, human review simply logs the request above.
                            </motion.p>
                          </CardContent>
                        </Card>
                      </motion.div>
                    );
                  default:
                    return null;
                }
              })()}
            </div>
          </div>
        </TransitionWrapper>

        {/* Footer note */}
        <motion.p
          className="mt-8 text-center text-xs text-graphite-600 font-mono"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1 }}
        >
          Frontend prototype · No backend, AI, or payment processing · All data simulated
        </motion.p>
      </main>

      {renderLightbox()}
    </div>
  );
}