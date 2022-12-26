import React, { useState, useEffect } from "react"

const RangeSlider = ({ changeFunction }) => {
    const [sliderValue, setSliderValue] = useState(0.0)

    const updateSlider = (event) => {
        setSliderValue(event.target.value)
    }

    useEffect(() => {
        changeFunction(sliderValue)
    }, [sliderValue])
    
    return (
        <div className="container">
            <div className="row">
                <div className="col-9">
                    <input className="form-range" type="range" value={sliderValue} min={0} max={1} step={0.1} onChange={updateSlider}/>
                </div>
                <div className="col-auto"> 
                    {sliderValue}
                </div>
            </div>
        </div>
    )
}

export default RangeSlider;