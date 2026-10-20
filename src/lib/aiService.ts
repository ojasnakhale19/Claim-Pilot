/**
 * AI Negotiation & Reasoning Assistant
 * Supports Gemini 1.5 Flash / OpenAI GPT-4o-mini / Local Fallback
 */

export interface AIAnalysisRequest {
  vehicle: {
    company: string;
    model: string;
    causeOfDamage: string;
  };
  damageSummary: string;
  severity: string;
  primaryComponent: string;
  costMin: number;
  costMax: number;
  settlementOffer: number;
  settlementMin: number;
  settlementMax: number;
  userMessage?: string;
  chatHistory?: Array<{ sender: 'user' | 'bot'; text: string }>;
  apiKey?: string;
  provider?: 'gemini' | 'openai' | 'rule-based';
}

export interface AIAnalysisResponse {
  message: string;
  newOffer?: number;
  isOffer?: boolean;
  recommendation?: string;
  reasoning?: string;
}

/**
 * Call Gemini Flash API for smart negotiation responses
 */
async function callGemini(
  prompt: string,
  apiKey: string
): Promise<string> {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${apiKey}`;
  
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      contents: [
        {
          parts: [{ text: prompt }]
        }
      ],
      generationConfig: {
        temperature: 0.3,
        maxOutputTokens: 500,
      }
    })
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Gemini API error: ${response.status} - ${errorText}`);
  }

  const data = await response.json();
  return data.candidates?.[0]?.content?.parts?.[0]?.text || '';
}

/**
 * Call OpenAI API (e.g. gpt-4o-mini)
 */
async function callOpenAI(
  systemPrompt: string,
  userMessage: string,
  apiKey: string
): Promise<string> {
  const url = 'https://api.openai.com/v1/chat/completions';
  
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: 'gpt-4o-mini',
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userMessage }
      ],
      temperature: 0.3,
      max_tokens: 500,
    })
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`OpenAI API error: ${response.status} - ${errorText}`);
  }

  const data = await response.json();
  return data.choices?.[0]?.message?.content || '';
}

/**
 * Get intelligent AI response combining local model stats with LLM reasoning
 */
export async function generateAINegotiationResponse(
  req: AIAnalysisRequest
): Promise<AIAnalysisResponse> {
  const apiKey = req.apiKey || localStorage.getItem('claimpilot_llm_key') || '';
  const provider = req.provider || (localStorage.getItem('claimpilot_llm_provider') as any) || (apiKey.startsWith('AIza') ? 'gemini' : apiKey.startsWith('sk-') ? 'openai' : 'rule-based');

  const systemContext = `You are ClaimPilot AI, a knowledgeable, professional motor insurance settlement adjuster assistant.
You are negotiating a fast cash settlement with a claimant for vehicle damage.
Vehicle: ${req.vehicle.company || 'Unknown'} ${req.vehicle.model || 'Vehicle'}.
Reported Incident: ${req.vehicle.causeOfDamage || 'Accident collision'}.
Assessed Damage by Computer Vision: ${req.damageSummary} (Severity: ${req.severity}, Component: ${req.primaryComponent}).
Estimated Garage Repair Cost: ₹${req.costMin.toLocaleString('en-IN')} - ₹${req.costMax.toLocaleString('en-IN')}.
Current Permitted Settlement Range: ₹${req.settlementMin.toLocaleString('en-IN')} to ₹${req.settlementMax.toLocaleString('en-IN')}.
Current Offer: ₹${req.settlementOffer.toLocaleString('en-IN')}.

Guidelines:
1. Always maintain a polite, transparent, and empathetic tone.
2. If the user asks for more money:
   - If they specify an amount <= ₹${req.settlementMax}, consider granting or counter-offering close to it.
   - If they specify an amount > ₹${req.settlementMax}, explain politely that insurance guidelines limit the maximum cash buyout to ₹${req.settlementMax.toLocaleString('en-IN')}, and offer that maximum.
   - If they just say "more" or "increase", offer an increment (e.g. +₹500 to +₹1,000) capped at ₹${req.settlementMax}.
3. If they ask why the cash offer is lower than repair cost: explain standard insurer settlement formulas (deductibles, retail labor vs negotiated rates, depreciation).
4. If they accept or agree: confirm clearly and prompt them to finalize.
5. Always keep replies concise (under 3 sentences).`;

  if (provider === 'gemini' && apiKey) {
    try {
      const fullPrompt = `${systemContext}\n\nClaimant said: "${req.userMessage || 'Hello, can you explain the settlement offer?'}"\n\nProvide your response. If you are proposing a specific offer amount, explicitly mention "₹[amount]".`;
      const responseText = await callGemini(fullPrompt, apiKey);

      // Extract offer if present
      const match = responseText.match(/₹\s?([\d,]+)/);
      let newOffer: number | undefined;
      if (match) {
        const val = parseInt(match[1].replace(/,/g, ''), 10);
        if (val >= req.settlementMin && val <= req.settlementMax) {
          newOffer = val;
        }
      }

      return {
        message: responseText.trim(),
        newOffer,
        isOffer: Boolean(newOffer),
      };
    } catch (err) {
      console.warn('Gemini API call failed, falling back to local reasoning:', err);
    }
  }

  if (provider === 'openai' && apiKey) {
    try {
      const responseText = await callOpenAI(
        systemContext,
        req.userMessage || 'Hello, can you explain the settlement offer?',
        apiKey
      );

      const match = responseText.match(/₹\s?([\d,]+)/);
      let newOffer: number | undefined;
      if (match) {
        const val = parseInt(match[1].replace(/,/g, ''), 10);
        if (val >= req.settlementMin && val <= req.settlementMax) {
          newOffer = val;
        }
      }

      return {
        message: responseText.trim(),
        newOffer,
        isOffer: Boolean(newOffer),
      };
    } catch (err) {
      console.warn('OpenAI API call failed, falling back to local reasoning:', err);
    }
  }

  // High quality local rule-based fallback (no API key needed)
  return getLocalIntelligentResponse(req);
}

function getLocalIntelligentResponse(req: AIAnalysisRequest): AIAnalysisResponse {
  const text = (req.userMessage || '').toLowerCase();
  const max = req.settlementMax;
  const current = req.settlementOffer;
  
  // Extract custom numbers
  const numMatch = text.match(/[₹$]?\s?(\d[\d,]*)/);
  const userNum = numMatch ? parseInt(numMatch[1].replace(/,/g, ''), 10) : null;

  if (text.includes('more') || text.includes('higher') || text.includes('increase') || text.includes('counter')) {
    if (userNum && userNum <= max) {
      return {
        message: `Based on your vehicle damage assessment, we can adjust the settlement offer to ₹${userNum.toLocaleString('en-IN')}. Would this amount be acceptable to proceed?`,
        newOffer: userNum,
        isOffer: true,
      };
    } else if (userNum && userNum > max) {
      return {
        message: `While we understand your request for ₹${userNum.toLocaleString('en-IN')}, our automated policy guidelines cap this component's instant payout at ₹${max.toLocaleString('en-IN')}. We can finalize at ₹${max.toLocaleString('en-IN')}.`,
        newOffer: max,
        isOffer: true,
      };
    } else {
      const incremented = Math.min(max, current + 600);
      return {
        message: `We can increase the instant settlement offer to ₹${incremented.toLocaleString('en-IN')} (authorized ceiling: ₹${max.toLocaleString('en-IN')}). Would you like to accept?`,
        newOffer: incremented,
        isOffer: true,
      };
    }
  }

  if (text.includes('why') || text.includes('lower') || text.includes('difference') || text.includes('explain')) {
    return {
      message: `The instant cash settlement (₹${current.toLocaleString('en-IN')}) provides immediate payout without garage bills or deductible delays. It factors standard parts depreciation and insurer-authorized volume labor rates rather than retail walk-in quotes.`,
      isOffer: false,
    };
  }

  if (text.includes('breakdown') || text.includes('parts') || text.includes('labor')) {
    const parts = Math.round(req.costMin * 0.45);
    const labor = Math.round(req.costMin * 0.35);
    const paint = req.costMin - parts - labor;
    return {
      message: `Estimated Assessment Breakdown:\n• Replacement / Parts: ₹${parts.toLocaleString('en-IN')}\n• Certified Labor: ₹${labor.toLocaleString('en-IN')}\n• Paint & Finishing: ₹${paint.toLocaleString('en-IN')}\nCash settlement reflects net payable after deductible exclusions.`,
      isOffer: false,
    };
  }

  if (text.includes('accept') || text.includes('agree') || text.includes('yes') || text.includes('ok') || text.includes('confirm')) {
    return {
      message: `Excellent! The settlement of ₹${current.toLocaleString('en-IN')} has been accepted. Please click 'Finalize Settlement' below to generate your claim confirmation.`,
      newOffer: current,
      isOffer: true,
    };
  }

  return {
    message: `Thank you for your update. The current settlement offer stands at ₹${current.toLocaleString('en-IN')} (authorized range: ₹${req.settlementMin.toLocaleString('en-IN')} - ₹${max.toLocaleString('en-IN')}). You can request a higher amount, ask for a cost breakdown, or accept this offer.`,
    isOffer: false,
  };
}
