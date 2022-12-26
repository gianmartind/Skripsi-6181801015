import React, { useEffect, useMemo, useState } from "react"


const ImagePicker = ({ changeImage }) => {
    const [image, setImage] = useState(null)

    const onImageChange = (event) => {
        setImage(event.target.files[0])
    }

    const image_src = useMemo(() => {
        if(image != null) {
            return URL.createObjectURL(image)
        } else {
            return '/img_placeholder.png'
        }
    }, [image])

    useEffect(() => {
        changeImage(image)
    }, [image])


    return (
        <div className="container p-0">
            <div className="container border rounded">
                <img src={image_src} className="img-fluid" />
            </div>
            <div className="container mt-1">
                <div className="border border-2 border-primary rounded p-1">
                    <input type="file" accept="image/*" onChange={onImageChange} />
                </div>
            </div>
        </div>
    )
}

export default ImagePicker